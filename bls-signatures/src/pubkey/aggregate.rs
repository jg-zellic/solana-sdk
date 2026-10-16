#[cfg(feature = "parallel")]
use rayon::prelude::*;
use {
    crate::{
        error::BlsError,
        pubkey::points::{AddToPubkeyProjective, AggregatePubkey, PopVerified, PubkeyProjective},
        scalar::Scalar,
    },
    blst::p1_affines,
    blstrs::Scalar as BlstrsScalar,
};

/// Bytes and bits blst reads per little-endian scalar.
const SCALAR_BYTES: usize = 32;
const SCALAR_BITS: usize = 255;

impl PubkeyProjective {
    /// Aggregate a list of Proof-of-Possession verified public keys into an
    /// existing aggregate.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn aggregate_with<'a, P: AddToPubkeyProjective + ?Sized + 'a>(
        &mut self,
        pubkeys: impl Iterator<Item = &'a PopVerified<P>>,
    ) -> Result<(), BlsError> {
        for pubkey in pubkeys {
            // Access the inner key via .0
            pubkey.0.add_to_accumulator(self)?;
        }
        Ok(())
    }

    /// Aggregate a list of Proof-of-Possession verified public keys.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn aggregate<'a, P: AddToPubkeyProjective + ?Sized + 'a>(
        pubkeys: impl Iterator<Item = &'a PopVerified<P>>,
    ) -> Result<AggregatePubkey<PubkeyProjective>, BlsError> {
        let mut aggregate = PubkeyProjective::identity();
        let mut count = 0;
        for pubkey in pubkeys {
            pubkey.0.add_to_accumulator(&mut aggregate)?;
            count += 1;
        }
        if count == 0 {
            return Err(BlsError::EmptyAggregation);
        }
        Ok(AggregatePubkey(aggregate))
    }

    /// Aggregate a list of Proof-of-Possession verified public keys with scalars.
    #[deprecated(
        since = "3.5.0",
        note = "Please use `PubkeyProjective::aggregate_with_weights` instead, which takes \
                `solana_bls_signatures::Scalar` rather than `blstrs::Scalar`"
    )]
    pub fn aggregate_with_scalars<'a, P: AddToPubkeyProjective + ?Sized + 'a>(
        pubkeys: impl ExactSizeIterator<Item = &'a PopVerified<P>>,
        scalars: impl ExactSizeIterator<Item = &'a BlstrsScalar>,
    ) -> Result<AggregatePubkey<PubkeyProjective>, BlsError> {
        Self::multi_exp(pubkeys, scalars.copied())
    }

    /// Aggregate a list of Proof-of-Possession verified public keys, each
    /// weighted by a scalar.
    pub fn aggregate_with_weights<'a, P: AddToPubkeyProjective + ?Sized + 'a>(
        pubkeys: impl ExactSizeIterator<Item = &'a PopVerified<P>>,
        weights: impl ExactSizeIterator<Item = &'a Scalar>,
    ) -> Result<AggregatePubkey<PubkeyProjective>, BlsError> {
        Self::multi_exp(pubkeys, weights.map(|weight| weight.0))
    }

    /// The multi-scalar multiplication shared by [`Self::aggregate_with_weights`]
    /// and its deprecated `blstrs`-typed counterpart.
    #[allow(clippy::arithmetic_side_effects)]
    fn multi_exp<'a, P: AddToPubkeyProjective + ?Sized + 'a>(
        pubkeys: impl ExactSizeIterator<Item = &'a PopVerified<P>>,
        scalars: impl ExactSizeIterator<Item = BlstrsScalar>,
    ) -> Result<AggregatePubkey<PubkeyProjective>, BlsError> {
        if pubkeys.len() != scalars.len() {
            return Err(BlsError::InputLengthMismatch);
        }

        if pubkeys.len() == 0 {
            return Err(BlsError::EmptyAggregation);
        }

        // Build the scalar bytes blst consumes directly, rather than a scalar
        // vector that `G1Projective::multi_exp` would copy into a second buffer.
        let mut points = alloc::vec::Vec::with_capacity(pubkeys.len());
        let mut scalar_bytes =
            alloc::vec::Vec::with_capacity(scalars.len().saturating_mul(SCALAR_BYTES));

        for (pubkey, scalar) in pubkeys.zip(scalars) {
            let mut point = PubkeyProjective::identity();
            pubkey.0.add_to_accumulator(&mut point)?;

            points.push(*point.0.as_ref());
            scalar_bytes.extend_from_slice(&scalar.to_bytes_le());
        }

        let mut aggregate = PubkeyProjective::identity();
        *aggregate.0.as_mut() = p1_affines::from(&points).mult(&scalar_bytes, SCALAR_BITS);
        Ok(AggregatePubkey(aggregate))
    }

    /// Aggregate a list of Proof-of-Possession verified public keys into an
    /// existing aggregate (Parallel)
    #[allow(clippy::arithmetic_side_effects)]
    #[cfg(feature = "parallel")]
    pub fn par_aggregate_with<'a, P: AddToPubkeyProjective + Sync + 'a>(
        &mut self,
        pubkeys: impl ParallelIterator<Item = &'a PopVerified<P>>,
    ) -> Result<(), BlsError> {
        match PubkeyProjective::par_aggregate(pubkeys) {
            Ok(aggregate) => {
                self.0 += &aggregate.0 .0;
                Ok(())
            }
            Err(BlsError::EmptyAggregation) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Aggregate a list of Proof-of-Possession verified public keys (Parallel)
    #[allow(clippy::arithmetic_side_effects)]
    #[cfg(feature = "parallel")]
    pub fn par_aggregate<'a, P: AddToPubkeyProjective + Sync + 'a>(
        pubkeys: impl ParallelIterator<Item = &'a PopVerified<P>>,
    ) -> Result<AggregatePubkey<PubkeyProjective>, BlsError> {
        let (aggregate, has_items) = pubkeys
            .into_par_iter()
            .fold(
                || Ok::<_, BlsError>((PubkeyProjective::identity(), false)),
                |acc, pubkey| {
                    let (mut proj, _) = acc?;
                    pubkey.0.add_to_accumulator(&mut proj)?;
                    Ok((proj, true))
                },
            )
            .reduce(
                || Ok::<_, BlsError>((PubkeyProjective::identity(), false)),
                |a, b| {
                    let (mut a_proj, a_has) = a?;
                    let (b_proj, b_has) = b?;
                    a_proj.0 += b_proj.0;
                    Ok((a_proj, a_has || b_has))
                },
            )?;

        if !has_items {
            return Err(BlsError::EmptyAggregation);
        }

        Ok(AggregatePubkey(aggregate))
    }
}

#[cfg(test)]
mod tests {
    use crate::{keypair::Keypair, pubkey::PubkeyProjective, scalar::Scalar};

    #[test]
    #[allow(deprecated)]
    fn test_deprecated_aggregate_with_scalars_matches_weights() {
        let pubkeys = [
            Keypair::new().public,
            Keypair::new().public,
            Keypair::new().public,
        ];
        let weights = [Scalar::random(), Scalar::random(), Scalar::random()];
        let blstrs_weights: alloc::vec::Vec<_> = weights.iter().map(|weight| weight.0).collect();

        let from_weights =
            PubkeyProjective::aggregate_with_weights(pubkeys.iter(), weights.iter()).unwrap();
        let from_scalars =
            PubkeyProjective::aggregate_with_scalars(pubkeys.iter(), blstrs_weights.iter())
                .unwrap();
        assert_eq!(from_weights, from_scalars);
    }

    #[test]
    fn test_aggregate_with_weights_matches_blstrs_multi_exp() {
        use blstrs::G1Projective;

        let keypairs = [Keypair::new(), Keypair::new(), Keypair::new()];
        // Cover blst's per-point path below 32 points and Pippenger above it.
        for n in [1, 2, 31, 32, 33, 64] {
            // Repeat keys and include zero and repeated weights.
            let pubkeys: alloc::vec::Vec<_> = (0..n).map(|i| keypairs[i % 3].public).collect();
            let weights: alloc::vec::Vec<_> = (0..n)
                .map(|i| match i % 4 {
                    0 => Scalar::from(0u64),
                    1 => Scalar::from(7u64),
                    _ => Scalar::random(),
                })
                .collect();

            let points: alloc::vec::Vec<_> = pubkeys
                .iter()
                .map(|pubkey| G1Projective::from(pubkey.0 .0))
                .collect();
            let scalars: alloc::vec::Vec<_> = weights.iter().map(|weight| weight.0).collect();
            let expected = G1Projective::multi_exp(&points, &scalars);

            let actual =
                PubkeyProjective::aggregate_with_weights(pubkeys.iter(), weights.iter()).unwrap();
            assert_eq!(actual.0 .0, expected, "n={n}");
        }
    }
}
