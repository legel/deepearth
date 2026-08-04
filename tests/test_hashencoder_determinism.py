import os
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class HashEncoderDeterminismTest(unittest.TestCase):
    def _gradients(self, precomputed):
        from deepearth.encoders.spacetime.hashencoder import HashEncoder

        torch.manual_seed(7)
        encoder = HashEncoder(
            input_dim=2,
            num_levels=4,
            level_dim=2,
            base_resolution=2,
            per_level_scale=2,
            log2_hashmap_size=4,
            enable_learned_probing=True,
            probing_range=2,
            index_codebook_size=8,
        ).cuda()
        inputs = torch.rand(4096, 2, device="cuda") * 2 - 1
        weights = torch.randn(4096, encoder.output_dim, device="cuda")
        if precomputed:
            encoder.precompute(inputs)
            output = encoder.forward_precomputed()
        else:
            output = encoder(inputs)
        (output * weights).sum().backward()
        return encoder.embeddings.grad.clone(), encoder.index_logits.grad.clone()

    def test_backward_is_bit_exact(self):
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for precomputed in (False, True):
                first = self._gradients(precomputed)
                second = self._gradients(precomputed)
                self.assertTrue(torch.equal(first[0], second[0]))
                self.assertTrue(torch.equal(first[1], second[1]))
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_backward_matches_float_atomics(self):
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            for precomputed in (False, True):
                torch.use_deterministic_algorithms(False)
                reference = self._gradients(precomputed)
                torch.use_deterministic_algorithms(True)
                candidate = self._gradients(precomputed)
                self.assertTrue(torch.allclose(reference[0], candidate[0], rtol=1e-4, atol=1e-5))
                self.assertTrue(torch.allclose(reference[1], candidate[1], rtol=1e-4, atol=1e-5))
        finally:
            torch.use_deterministic_algorithms(previous)


if __name__ == "__main__":
    unittest.main()
