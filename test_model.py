"""
Automated Verification Test Suite for All-Token Recurrence Architecture.

Verifies:
1. Tensor dimension consistency across varying sequence lengths and batch sizes.
2. Step-by-step state transition: verifies recurrent state H_T and KV cache correctly
   update and maintain historical context.
3. Gradient backpropagation: ensures all parameter weights receive non-zero, finite
   gradients without numerical instabilities (NaN or Inf).
4. Causal masking and generation sanity.
"""

import pytest
import torch
import torch.nn.functional as F

from model import AllTokenRecurrentConfig, AllTokenRecurrentModel


@pytest.fixture
def base_config():
    """Returns a deterministic small config suitable for fast test verification."""
    return AllTokenRecurrentConfig(
        vocab_size=32,
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_seq_len=128,
        dropout=0.0,
        tie_weights=False,
    )


class TestTensorDimensions:
    """R2.1: Tensor dimension consistency across varying sequence lengths and batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 7])
    @pytest.mark.parametrize("seq_len", [1, 2, 5, 13, 27])
    def test_forward_dimensions(self, base_config, batch_size, seq_len):
        torch.manual_seed(42)
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))
        logits = model(input_ids)

        assert isinstance(logits, torch.Tensor), "Output should be a torch.Tensor"
        assert logits.shape == (batch_size, seq_len, base_config.vocab_size), (
            f"Expected logits shape {(batch_size, seq_len, base_config.vocab_size)}, got {logits.shape}"
        )
        assert not torch.isnan(logits).any(), "Forward pass produced NaN logits"
        assert not torch.isinf(logits).any(), "Forward pass produced Inf logits"

    def test_return_state_dimensions(self, base_config):
        torch.manual_seed(42)
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        batch_size, seq_len = 3, 8
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))
        logits, final_state, (prefix_kv, dec_kv) = model(input_ids, return_state=True)

        # Verify logits shape
        assert logits.shape == (batch_size, seq_len, base_config.vocab_size)

        # Verify final recurrent state H_T shape: (B, d_model)
        assert final_state.shape == (batch_size, base_config.d_model)

        # Verify prefix KV memory shape for each encoder layer
        assert len(prefix_kv) == base_config.num_encoder_layers
        d_head = base_config.d_model // base_config.num_heads
        for pk, pv in prefix_kv:
            assert pk.shape == (batch_size, base_config.num_heads, seq_len, d_head)
            assert pv.shape == (batch_size, base_config.num_heads, seq_len, d_head)

        # Verify decoder KV cache shape for each decoder layer
        assert len(dec_kv) == base_config.num_decoder_layers
        for dk, dv in dec_kv:
            assert dk.shape == (batch_size, base_config.num_heads, seq_len, d_head)
            assert dv.shape == (batch_size, base_config.num_heads, seq_len, d_head)

    def test_empty_sequence_forward(self, base_config):
        """Edge case: sequence length 0 returns empty tensor without crashing."""
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        input_ids = torch.empty((2, 0), dtype=torch.long)
        logits = model(input_ids)
        assert logits.shape == (2, 0, base_config.vocab_size)

        logits, final_state, (prefix_kv, dec_kv) = model(input_ids, return_state=True)
        assert logits.shape == (2, 0, base_config.vocab_size)
        assert final_state.shape == (2, base_config.d_model)

    def test_unequal_encoder_decoder_layers(self):
        """Verifies architecture functions properly when L_E != L_D."""
        # Case A: L_E = 1, L_D = 3
        cfg_a = AllTokenRecurrentConfig(
            vocab_size=16, d_model=16, num_heads=2, d_ff=32,
            num_encoder_layers=1, num_decoder_layers=3
        )
        model_a = AllTokenRecurrentModel(cfg_a)
        out_a = model_a(torch.randint(0, 16, (2, 4)))
        assert out_a.shape == (2, 4, 16)

        # Case B: L_E = 3, L_D = 1
        cfg_b = AllTokenRecurrentConfig(
            vocab_size=16, d_model=16, num_heads=2, d_ff=32,
            num_encoder_layers=3, num_decoder_layers=1
        )
        model_b = AllTokenRecurrentModel(cfg_b)
        out_b = model_b(torch.randint(0, 16, (2, 4)))
        assert out_b.shape == (2, 4, 16)

    def test_single_head_attention(self):
        """Verifies architecture functions properly with single attention head (num_heads=1)."""
        cfg = AllTokenRecurrentConfig(
            vocab_size=16, d_model=16, num_heads=1, d_ff=32,
            num_encoder_layers=1, num_decoder_layers=1
        )
        model = AllTokenRecurrentModel(cfg)
        out = model(torch.randint(0, 16, (2, 4)))
        assert out.shape == (2, 4, 16)

    def test_custom_initial_state(self, base_config):
        """Verifies forward pass supports user-provided initial recurrent state."""
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        batch_size, seq_len = 2, 4
        custom_h0 = torch.randn(batch_size, base_config.d_model)
        logits = model(torch.randint(0, base_config.vocab_size, (batch_size, seq_len)), initial_state=custom_h0)
        assert logits.shape == (batch_size, seq_len, base_config.vocab_size)


class TestStepByStepStateTransition:
    """R2.2: Step-by-step state transition, KV cache updates, and historical context."""

    def test_step_by_step_numerical_equivalence(self, base_config):
        """
        Verifies that running step() sequentially token-by-token produces
        mathematically identical logits and recurrent states to forward().
        """
        torch.manual_seed(1337)
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        batch_size, seq_len = 2, 6
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))

        # 1. Full sequence forward
        logits_full, h_full, (prefix_full, dec_full) = model(input_ids, return_state=True)

        # 2. Sequential step-by-step execution
        h = model.init_recurrent_state(batch_size)
        kv_cache = None
        step_logits_list = []

        for t in range(seq_len):
            token_t = input_ids[:, t:t + 1]
            logit_t, h, kv_cache = model.step(token_t, h, kv_cache)
            step_logits_list.append(logit_t)

            # Check KV cache length at step t
            prefix_kv, dec_kv = kv_cache
            for pk, pv in prefix_kv:
                assert pk.size(2) == t + 1
                assert pv.size(2) == t + 1
            for dk, dv in dec_kv:
                assert dk.size(2) == t + 1
                assert dv.size(2) == t + 1

        logits_step = torch.cat(step_logits_list, dim=1)

        # Verify numerical equivalence
        max_logits_diff = (logits_full - logits_step).abs().max().item()
        assert max_logits_diff < 1e-5, f"Logits discrepancy between full and step mode: {max_logits_diff}"

        max_state_diff = (h_full - h).abs().max().item()
        assert max_state_diff < 1e-5, f"Final recurrent state discrepancy: {max_state_diff}"

    def test_recurrent_state_evolution(self, base_config):
        """Verifies that recurrent state H updates non-trivially at each step (H_t != H_{t-1})."""
        torch.manual_seed(42)
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        batch_size, seq_len = 2, 4
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))

        h = model.init_recurrent_state(batch_size)
        kv_cache = None
        past_states = [h.clone()]

        for t in range(seq_len):
            token_t = input_ids[:, t:t + 1]
            _, h, kv_cache = model.step(token_t, h, kv_cache)
            # The state must transition and not stay identical
            state_diff = (h - past_states[-1]).abs().max().item()
            assert state_diff > 1e-4, f"Recurrent state did not update at step {t} (diff={state_diff})"
            past_states.append(h.clone())

    def test_historical_context_maintenance(self, base_config):
        """
        Verifies that recurrent state H_T and downstream predictions maintain historical context.
        Perturbing an early prompt token x_0 must change the final recurrent state H_T.
        """
        torch.manual_seed(99)
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        batch_size, seq_len = 2, 6
        x1 = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))
        x2 = x1.clone()
        # Perturb only the very first token
        x2[:, 0] = (x2[:, 0] + 5) % base_config.vocab_size

        _, h1, _ = model(x1, return_state=True)
        _, h2, _ = model(x2, return_state=True)

        context_diff = (h1 - h2).abs().max().item()
        assert context_diff > 1e-3, (
            f"Perturbing prompt token at position 0 had no effect on H_T (diff={context_diff}). "
            "Historical context is not maintained!"
        )


class TestGradientBackpropagation:
    """R2.3: Gradient backpropagation, finite non-zero gradients across all weights."""

    def test_backward_populates_all_gradients_cross_entropy(self, base_config):
        """
        Performs backward pass on synthetic cross-entropy next-token loss and verifies
        that ALL model parameter weights receive non-zero, finite gradients.
        """
        torch.manual_seed(101)
        model = AllTokenRecurrentModel(base_config)
        model.train()

        batch_size, seq_len = 2, base_config.vocab_size
        # Synthetic tokens covering vocabulary to ensure all embedding rows are indexed
        input_ids = torch.arange(base_config.vocab_size).repeat(batch_size, 1)[:, :seq_len]
        targets = (input_ids + 1) % base_config.vocab_size

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, base_config.vocab_size), targets.view(-1))

        loss.backward()

        # Check every named parameter
        assert len(list(model.parameters())) > 0, "Model has no parameters"

        for name, param in model.named_parameters():
            # 1. Gradient must exist
            assert param.grad is not None, f"Parameter '{name}' did not receive any gradient (is None)"

            # 2. No NaN gradients
            assert not torch.isnan(param.grad).any(), f"Parameter '{name}' contains NaN in gradients"

            # 3. No Inf gradients
            assert not torch.isinf(param.grad).any(), f"Parameter '{name}' contains Inf in gradients"

            # 4. Gradient must be non-zero (non-trivial)
            grad_norm = param.grad.norm().item()
            assert grad_norm > 0.0, f"Parameter '{name}' received an all-zero gradient"
            assert (param.grad != 0).any(), f"Parameter '{name}' has all zero values in gradient tensor"

    def test_backward_populates_both_encoder_and_decoder(self, base_config):
        """Explicitly checks that both encoder and recurrent decoder submodules receive gradients."""
        torch.manual_seed(202)
        model = AllTokenRecurrentModel(base_config)
        model.train()

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()

        # Verify encoder parameters
        encoder_params = [p for n, p in model.encoder.named_parameters()]
        assert len(encoder_params) > 0
        for p in encoder_params:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()
            assert (p.grad != 0).any()

        # Verify decoder parameters
        decoder_params = [p for n, p in model.decoder.named_parameters()]
        assert len(decoder_params) > 0
        for p in decoder_params:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()
            assert (p.grad != 0).any()

        # Verify transition parameters and h0
        assert model.decoder.h0.grad is not None and (model.decoder.h0.grad != 0).any()
        assert model.decoder.transition_proj.weight.grad is not None and (model.decoder.transition_proj.weight.grad != 0).any()

    def test_backward_with_tied_weights(self):
        """Verifies backward pass works when embedding and lm_head weights are tied."""
        torch.manual_seed(303)
        config = AllTokenRecurrentConfig(
            vocab_size=32,
            d_model=32,
            num_heads=2,
            d_ff=64,
            num_encoder_layers=1,
            num_decoder_layers=1,
            tie_weights=True,
        )
        model = AllTokenRecurrentModel(config)
        model.train()

        input_ids = torch.randint(0, config.vocab_size, (2, 4))
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"Tied weight {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Tied weight {name} has NaN gradient"
            assert (param.grad != 0).any(), f"Tied weight {name} has zero gradient"


class TestCausalityAndGeneration:
    """Additional architectural integrity checks: causality and generation."""

    def test_causal_masking_integrity(self, base_config):
        """
        Verifies that modifying future tokens x_{t+1} does NOT alter logits at step t.
        This guarantees zero forward information leakage across time steps.
        """
        torch.manual_seed(404)
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        batch_size, seq_len = 2, 5
        x1 = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))
        x2 = x1.clone()
        # Change token at position 3 and 4 only
        x2[:, 3:] = (x2[:, 3:] + 7) % base_config.vocab_size

        logits1 = model(x1)
        logits2 = model(x2)

        # Positions 0, 1, 2 must be IDENTICAL
        for t in range(3):
            diff = (logits1[:, t, :] - logits2[:, t, :]).abs().max().item()
            assert diff < 1e-5, f"Causal violation at step {t}: future token modified past logits (diff={diff})"

        # Position 3 and 4 should differ
        diff_future = (logits1[:, 3:, :] - logits2[:, 3:, :]).abs().max().item()
        assert diff_future > 1e-3, "Future modification did not change future logits"

    def test_generation(self, base_config):
        """Verifies autoregressive generation loop."""
        torch.manual_seed(505)
        model = AllTokenRecurrentModel(base_config)

        prompt = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        gen = model.generate(prompt, max_new_tokens=4, temperature=0.0)

        assert gen.shape == (2, 7)
        # First 3 tokens must match prompt
        assert torch.equal(gen[:, :3], prompt)
        # Generated tokens must be valid vocab indices
        assert (gen >= 0).all() and (gen < base_config.vocab_size).all()

    def test_generation_max_new_tokens_zero(self, base_config):
        """Edge case: max_new_tokens <= 0 returns original prompt without generating extra tokens."""
        model = AllTokenRecurrentModel(base_config)
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        gen0 = model.generate(prompt, max_new_tokens=0)
        assert torch.equal(gen0, prompt), f"Expected unchanged prompt shape (1, 3), got {gen0.shape}"

    def test_generation_temperature_stochasticity(self, base_config):
        """Verifies that temperature > 0 correctly samples stochastically on the very first token."""
        torch.manual_seed(1234)
        model = AllTokenRecurrentModel(base_config)
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

        # High temperature should yield diverse tokens even with max_new_tokens=1
        sampled_tokens = set()
        for _ in range(30):
            gen = model.generate(prompt, max_new_tokens=1, temperature=5.0)
            sampled_tokens.add(gen[0, -1].item())

        assert len(sampled_tokens) > 1, "Temperature > 0 produced deterministic first token"

        # Temperature = 0 should be strictly deterministic
        det_tokens = set()
        for _ in range(10):
            gen = model.generate(prompt, max_new_tokens=1, temperature=0.0)
            det_tokens.add(gen[0, -1].item())
        assert len(det_tokens) == 1, "Temperature = 0 produced non-deterministic output"


class TestMixedPrecisionAndRobustness:
    """Verifies low-precision dtypes, autocast, long sequences, and accelerator device execution."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtypes_forward_backward_step(self, base_config, dtype):
        """Verifies model forward, backward, and step passes cleanly under float32, float16, and bfloat16."""
        torch.manual_seed(777)
        model = AllTokenRecurrentModel(base_config).to(dtype=dtype)
        model.train()

        x = torch.randint(0, base_config.vocab_size, (2, 6))
        logits, h, kv_cache = model(x, return_state=True)
        assert logits.dtype == dtype
        assert h.dtype == dtype

        loss = logits.sum()
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name} with dtype {dtype}"
                assert not torch.isinf(p.grad).any(), f"Inf gradient in {name} with dtype {dtype}"

        # Test step transition in this dtype
        model.eval()
        single_tok = torch.randint(0, base_config.vocab_size, (2, 1))
        step_logits, step_h, _ = model.step(single_tok, h.detach(), kv_cache)
        assert step_logits.dtype == dtype
        assert step_h.dtype == dtype

    def test_autocast_cpu(self, base_config):
        """Verifies forward and backward execution under torch.autocast for CPU."""
        model = AllTokenRecurrentModel(base_config)
        x = torch.randint(0, base_config.vocab_size, (2, 6))

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(x)
            loss = logits.sum()
        loss.backward()
        assert not torch.isnan(model.encoder.layers[0].self_attn.q_proj.weight.grad).any()

    def test_accelerator_mps(self, base_config):
        """Verifies model executes cleanly on MPS accelerator if available."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS device not available on this platform")

        model = AllTokenRecurrentModel(base_config).to("mps")
        x = torch.randint(0, base_config.vocab_size, (2, 4), device="mps")

        logits = model(x)
        assert logits.device.type == "mps"
        loss = logits.sum()
        loss.backward()

        h = model.init_recurrent_state(2, device=torch.device("mps"))
        step_logits, step_h, _ = model.step(x[:, :1], h)
        assert step_logits.device.type == "mps"

    def test_long_sequence_stability(self, base_config):
        """Verifies long sequence forward and backward pass completes without numerical instability."""
        torch.manual_seed(888)
        cfg = AllTokenRecurrentConfig(
            vocab_size=32, d_model=32, num_heads=2, d_ff=64,
            num_encoder_layers=2, num_decoder_layers=2, max_seq_len=256
        )
        model = AllTokenRecurrentModel(cfg)
        model.train()

        x = torch.randint(0, 32, (1, 128))
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None
            assert not torch.isnan(p.grad).any(), f"NaN gradient in long sequence: {name}"
            assert not torch.isinf(p.grad).any(), f"Inf gradient in long sequence: {name}"


class TestInputFlexibilityAndBroadcasting:
    """Verifies API robustness with 1D inputs, scalar tokens, and state broadcasting."""

    def test_1d_input_forward(self, base_config):
        """Verifies forward pass handles 1D token tensor (seq_len,) gracefully."""
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        x_1d = torch.randint(0, base_config.vocab_size, (8,))
        logits = model(x_1d)
        assert logits.shape == (1, 8, base_config.vocab_size)

    def test_scalar_token_step(self, base_config):
        """Verifies step() handles scalar 0-D token tensor gracefully."""
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        h0 = model.init_recurrent_state(1)
        scalar_tok = torch.tensor(3)
        logits, h1, cache = model.step(scalar_tok, h0)
        assert logits.shape == (1, 1, base_config.vocab_size)
        assert h1.shape == (1, base_config.d_model)

    def test_broadcast_initial_state(self, base_config):
        """Verifies passing (1, d_model) initial state broadcasts to batch_size > 1."""
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        batch_size, seq_len = 3, 5
        h_single = torch.randn(1, base_config.d_model)
        x = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))

        # Forward pass with broadcast initial state
        logits, h_final, cache = model(x, initial_state=h_single, return_state=True)
        assert logits.shape == (batch_size, seq_len, base_config.vocab_size)
        assert h_final.shape == (batch_size, base_config.d_model)

        # Step transition with broadcast initial state
        tok = torch.randint(0, base_config.vocab_size, (batch_size, 1))
        step_logits, step_h, _ = model.step(tok, recurrent_state=h_single)
        assert step_logits.shape == (batch_size, 1, base_config.vocab_size)
        assert step_h.shape == (batch_size, base_config.d_model)

    @pytest.mark.parametrize("L_E, L_D", [(1, 3), (3, 1)])
    def test_unequal_layers_step_equivalence(self, L_E, L_D):
        """Verifies step-by-step equivalence when num_encoder_layers != num_decoder_layers."""
        torch.manual_seed(999)
        cfg = AllTokenRecurrentConfig(
            vocab_size=16, d_model=16, num_heads=2, d_ff=32,
            num_encoder_layers=L_E, num_decoder_layers=L_D
        )
        model = AllTokenRecurrentModel(cfg)
        model.eval()

        x = torch.randint(0, 16, (2, 5))
        logits_full, h_full, _ = model(x, return_state=True)

        h = model.init_recurrent_state(2)
        cache = None
        step_logits = []
        for t in range(5):
            logit_t, h, cache = model.step(x[:, t:t + 1], h, cache)
            step_logits.append(logit_t)
        logits_step = torch.cat(step_logits, dim=1)

        diff = (logits_full - logits_step).abs().max().item()
        assert diff < 1e-5, f"Step equivalence mismatch for L_E={L_E}, L_D={L_D}: diff={diff}"

    def test_scalar_input_forward(self, base_config):
        """Verifies forward pass accepts 0-D scalar token tensor."""
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        scalar_x = torch.tensor(7)
        logits = model(scalar_x)
        assert logits.shape == (1, 1, base_config.vocab_size)
        assert not torch.isnan(logits).any()

    def test_generation_1d_and_scalar_prompt(self, base_config):
        """Verifies generate() accepts 1-D prompts and 0-D scalar prompts, preserving rank."""
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        # 1-D prompt
        prompt_1d = torch.tensor([3, 5, 7])
        gen_1d = model.generate(prompt_1d, max_new_tokens=3, temperature=0.0)
        assert gen_1d.ndim == 1
        assert gen_1d.shape == (6,)
        assert torch.equal(gen_1d[:3], prompt_1d)

        # 0-D scalar prompt
        prompt_0d = torch.tensor(5)
        gen_0d = model.generate(prompt_0d, max_new_tokens=2, temperature=0.0)
        assert gen_0d.ndim == 1
        assert gen_0d.shape == (3,)
        assert gen_0d[0].item() == 5

    def test_generation_empty_prompt(self, base_config):
        """Verifies generate() can generate continuation tokens from empty prompt via H_0."""
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        # 2-D empty prompt
        prompt_empty_2d = torch.empty((2, 0), dtype=torch.long)
        gen_2d = model.generate(prompt_empty_2d, max_new_tokens=4, temperature=0.0)
        assert gen_2d.shape == (2, 4)
        assert (gen_2d >= 0).all() and (gen_2d < base_config.vocab_size).all()

        # 1-D empty prompt
        prompt_empty_1d = torch.empty((0,), dtype=torch.long)
        gen_1d = model.generate(prompt_empty_1d, max_new_tokens=3, temperature=0.0)
        assert gen_1d.shape == (3,)
        assert (gen_1d >= 0).all() and (gen_1d < base_config.vocab_size).all()

    def test_state_dtype_and_device_coercion(self, base_config):
        """Verifies that user-supplied initial_state and recurrent_state are automatically coerced in dtype."""
        model = AllTokenRecurrentModel(base_config).half()
        model.eval()

        # User passes float32 initial state to half-precision model
        h0_float32 = torch.zeros(2, base_config.d_model, dtype=torch.float32)
        x = torch.randint(0, base_config.vocab_size, (2, 4))
        logits, h_final, kv = model(x, initial_state=h0_float32, return_state=True)
        assert logits.dtype == torch.float16
        assert h_final.dtype == torch.float16

        # User passes float32 recurrent state to half-precision step
        tok = torch.tensor([[1], [2]])
        step_logits, step_h, _ = model.step(tok, recurrent_state=h0_float32)
        assert step_logits.dtype == torch.float16
        assert step_h.dtype == torch.float16

    def test_step_and_generation_input_validation(self, base_config):
        """Verifies error handling for invalid input dimensions and out-of-range parameters."""
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        h0 = model.init_recurrent_state(2)

        # Multi-token tensor passed to step() should assert
        multi_tok = torch.randint(0, base_config.vocab_size, (2, 3))
        with pytest.raises(AssertionError, match="step\\(\\) expects single-token input"):
            model.step(multi_tok, h0)

        # Negative step_idx in step() should assert
        tok = torch.tensor([[1], [2]])
        with pytest.raises(AssertionError, match="step_idx must be >= 0"):
            model.step(tok, h0, step_idx=-1)

        # Mismatched recurrent state batch size should assert
        h_mismatch = torch.randn(3, base_config.d_model)
        with pytest.raises(AssertionError, match="batch size"):
            model.step(tok, h_mismatch)

        # Negative temperature in generate() should assert
        with pytest.raises(AssertionError, match="temperature must be non-negative"):
            model.generate(tok, max_new_tokens=2, temperature=-1.0)

    @pytest.mark.parametrize("invalid_cfg", [
        {"d_model": -8},
        {"num_heads": 0},
        {"d_ff": 0},
        {"vocab_size": 0},
        {"d_model": 15, "num_heads": 4},  # not divisible
        {"num_encoder_layers": 0},
        {"num_decoder_layers": -1},
    ])
    def test_invalid_config_validation(self, invalid_cfg):
        """Verifies configuration constraints fail fast with informative assertion errors."""
        kwargs = {
            "vocab_size": 16, "d_model": 16, "num_heads": 2, "d_ff": 32,
            "num_encoder_layers": 1, "num_decoder_layers": 1,
        }
        kwargs.update(invalid_cfg)
        cfg = AllTokenRecurrentConfig(**kwargs)
        with pytest.raises(AssertionError):
            AllTokenRecurrentModel(cfg)

    def test_unrolled_step_backward_equivalence(self, base_config):
        """
        Verifies that unrolling step() in training mode produces mathematically identical
        forward outputs and backward gradients to the parallel forward() pass.
        """
        import copy

        torch.manual_seed(42)
        model1 = AllTokenRecurrentModel(base_config)
        model2 = copy.deepcopy(model1)
        model1.train()
        model2.train()

        tokens = torch.randint(0, base_config.vocab_size, (2, 6))

        # Forward pass 1: full model
        out1 = model1(tokens)
        loss1 = out1.sum()
        loss1.backward()

        # Forward pass 2: unrolled step()
        h = model2.init_recurrent_state(2)
        cache = None
        outs2 = []
        for t in range(6):
            logit_t, h, cache = model2.step(tokens[:, t:t + 1], h, cache)
            outs2.append(logit_t)
        out2 = torch.cat(outs2, dim=1)
        loss2 = out2.sum()
        loss2.backward()

        # Forward equivalence
        assert (out1 - out2).abs().max().item() < 1e-5

        # Gradient equivalence across all parameters
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            diff = (p1.grad - p2.grad).abs().max().item()
            assert diff < 1e-4, f"Gradient mismatch in parameter {n1}: max diff={diff}"


class TestReviewerHardeningAndEdgeCases:
    """Additional edge cases uncovered during adversarial review."""

    def test_kv_cache_batch_broadcast_in_step(self, base_config):
        """
        Verifies that when token_id has batch_size B > 1, a kv_cache with batch_size 1
        (e.g., from encoding a single prompt for beam search or multi-candidate sampling)
        is automatically broadcast to B without dimension mismatch crashes.
        """
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        # Encode single prompt (B=1, T=3)
        prompt = torch.tensor([[1, 2, 3]])
        _, h1, kv_cache = model(prompt, return_state=True)
        assert h1.shape == (1, base_config.d_model)

        # Step with candidate tokens from B=3 branches
        cand_tokens = torch.tensor([[4], [5], [6]])  # B=3
        logits_step, h_step, new_cache = model.step(cand_tokens, recurrent_state=h1, kv_cache=kv_cache)

        assert logits_step.shape == (3, 1, base_config.vocab_size)
        assert h_step.shape == (3, base_config.d_model)
        pref_mem, dec_cache = new_cache
        for pk, pv in pref_mem:
            assert pk.shape == (3, base_config.num_heads, 4, base_config.d_model // base_config.num_heads)
            assert pv.shape == (3, base_config.num_heads, 4, base_config.d_model // base_config.num_heads)
        for dk, dv in dec_cache:
            assert dk.shape == (3, base_config.num_heads, 4, base_config.d_model // base_config.num_heads)
            assert dv.shape == (3, base_config.num_heads, 4, base_config.d_model // base_config.num_heads)

        # Ensure second step also works smoothly on the updated cache
        cand_tokens_2 = torch.tensor([[7], [8], [9]])
        logits_step2, h_step2, new_cache2 = model.step(cand_tokens_2, recurrent_state=h_step, kv_cache=new_cache)
        assert logits_step2.shape == (3, 1, base_config.vocab_size)
        assert h_step2.shape == (3, base_config.d_model)

    def test_kv_cache_dtype_and_device_coercion(self, base_config):
        """
        Verifies that passing float32 kv_cache into half-precision model step()
        is coerced properly without scalar type mismatch errors.
        """
        model = AllTokenRecurrentModel(base_config).half()
        model.eval()

        prompt = torch.tensor([[1, 2]])
        _, h, kv = model(prompt, return_state=True)

        # Convert cache to float32
        kv_f32 = (
            [(k.float(), v.float()) for k, v in kv[0]],
            [(k.float(), v.float()) for k, v in kv[1]],
        )
        tok = torch.tensor([[3]])
        logits, new_h, new_kv = model.step(tok, h, kv_f32)
        assert logits.dtype == torch.float16
        assert new_h.dtype == torch.float16

    def test_cpu_inputs_on_mps_device(self, base_config):
        """
        Verifies that CPU input tensors passed to forward(), step(), and generate()
        on an MPS model are automatically aligned with the model device without crashing.
        """
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        model = AllTokenRecurrentModel(base_config).to("mps")
        model.eval()

        # 1. CPU input_ids in forward
        cpu_x = torch.tensor([[1, 2, 3]])
        out = model(cpu_x)
        assert out.device.type == "mps"

        # 2. CPU token_id in step
        cpu_tok = torch.tensor([[4]])
        step_out, step_h, _ = model.step(cpu_tok)
        assert step_out.device.type == "mps"
        assert step_h.device.type == "mps"

        # 3. CPU prompt_ids in generate
        gen = model.generate(cpu_x, max_new_tokens=3, temperature=0.0)
        assert gen.device.type == "mps"
        assert gen.shape == (1, 6)

    def test_empty_sequence_backward(self, base_config):
        """
        Verifies that forward pass on an empty sequence maintains autograd connectivity
        and loss.backward() executes cleanly without RuntimeError.
        """
        model = AllTokenRecurrentModel(base_config)
        model.train()

        empty_ids = torch.empty((2, 0), dtype=torch.long)
        logits = model(empty_ids)
        assert logits.shape == (2, 0, base_config.vocab_size)
        loss = logits.sum()
        loss.backward()

    def test_generate_3d_input_validation(self, base_config):
        """Verifies that passing a 3D tensor to generate() raises an informative AssertionError."""
        model = AllTokenRecurrentModel(base_config)
        prompt_3d = torch.randint(0, base_config.vocab_size, (2, 3, 4))
        with pytest.raises(AssertionError, match="prompt_ids must have at most 2 dimensions"):
            model.generate(prompt_3d)

    def test_generate_with_custom_initial_state(self, base_config):
        """Verifies generate() accepts user-provided initial_state for non-empty and empty prompts."""
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        custom_h = torch.randn(2, base_config.d_model)

        # Non-empty prompt
        prompt = torch.tensor([[1, 2], [3, 4]])
        gen = model.generate(prompt, max_new_tokens=3, initial_state=custom_h, temperature=0.0)
        assert gen.shape == (2, 5)
        assert torch.equal(gen[:, :2], prompt)

        # Empty prompt
        empty_prompt = torch.empty((2, 0), dtype=torch.long)
        gen_empty = model.generate(empty_prompt, max_new_tokens=3, initial_state=custom_h, temperature=0.0)
        assert gen_empty.shape == (2, 3)

    def test_step_with_none_recurrent_state(self, base_config):
        """Verifies step() defaults to init_recurrent_state when recurrent_state is None."""
        model = AllTokenRecurrentModel(base_config)
        model.eval()

        tok = torch.tensor([[1], [2]])
        logits, h, cache = model.step(tok, recurrent_state=None)
        assert logits.shape == (2, 1, base_config.vocab_size)
        assert h.shape == (2, base_config.d_model)

    def test_layer_count_mismatch_validation(self, base_config):
        """Verifies clear assertions when past cache has mismatched number of layers."""
        model = AllTokenRecurrentModel(base_config)

        # past_prefix_kv layer count mismatch
        wrong_prefix_kv = [None] * (base_config.num_encoder_layers + 1)
        with pytest.raises(AssertionError, match="past_prefix_kv has"):
            model.encoder(torch.randn(1, 2, base_config.d_model), past_prefix_kv=wrong_prefix_kv)

        # decoder_kv_cache layer count mismatch
        wrong_dec_kv = [None] * (base_config.num_decoder_layers + 1)
        _, _, (pref, _) = model(torch.tensor([[1, 2]]), return_state=True)
        with pytest.raises(AssertionError, match="decoder_kv_cache has"):
            model.decoder.step(
                h_prev=torch.randn(1, base_config.d_model),
                e_t=torch.randn(1, base_config.d_model),
                decoder_kv_cache=wrong_dec_kv,
                prefix_kv_memory=pref,
                step_idx=0,
            )

    def test_cache_batch_size_mismatch_assertion(self, base_config):
        """Verifies assertion error when cache batch size is incompatible with token batch size."""
        model = AllTokenRecurrentModel(base_config)
        tok = torch.tensor([[1], [2]])  # B=2

        d_head = base_config.d_model // base_config.num_heads
        # Cache with B=3 (neither matches B=2 nor is 1)
        mismatched_k = torch.randn(3, base_config.num_heads, 2, d_head)
        mismatched_v = torch.randn(3, base_config.num_heads, 2, d_head)
        mismatched_cache = ([(mismatched_k, mismatched_v)] * base_config.num_encoder_layers, None)

        with pytest.raises(AssertionError, match="prefix_kv_memory batch size 3 does not match token batch size 2"):
            model.step(tok, kv_cache=mismatched_cache)

    @pytest.mark.parametrize("invalid_kwargs", [
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"dropout": 1.5},
        {"max_seq_len": 0},
        {"max_seq_len": -10},
    ])
    def test_extended_invalid_config_validation(self, invalid_kwargs):
        """Verifies assertions on invalid dropout and max_seq_len parameters."""
        kwargs = {
            "vocab_size": 16, "d_model": 16, "num_heads": 2, "d_ff": 32,
            "num_encoder_layers": 1, "num_decoder_layers": 1,
        }
        kwargs.update(invalid_kwargs)
        cfg = AllTokenRecurrentConfig(**kwargs)
        with pytest.raises(AssertionError):
            AllTokenRecurrentModel(cfg)

    def test_synthetic_training_convergence(self):
        """
        Verifies actual loss convergence on a deterministic sequence prediction task.
        Ensures model architecture is trainable and gradients drive effective optimization.
        """
        torch.manual_seed(42)
        cfg = AllTokenRecurrentConfig(
            vocab_size=16, d_model=32, num_heads=2, d_ff=64,
            num_encoder_layers=1, num_decoder_layers=1,
        )
        model = AllTokenRecurrentModel(cfg)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

        x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        y = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 1]])

        initial_loss = None
        final_loss = None

        for step in range(30):
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        assert final_loss < initial_loss * 0.25, (
            f"Expected loss to decrease significantly, but initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

