"""Parameter-count regression tests for the GDN-2 host model.

`compute_model_params` analytically predicts the model's parameter count. This
suite guards it against drifting away from the actual ``nn.Module`` structure
(e.g. the value-side projection stack using per-head vs flat dims).
"""

from qwendopamine.models.surprise_gpt.config import SurpriseGPTConfig
from qwendopamine.models.surprise_gpt.model import (
    SurpriseGPT,
    compute_model_params,
)


def test_when_gdn2_only_model_then_param_counts_match_reality() -> None:
    cfg = SurpriseGPTConfig.from_name(
        "1B_mha",
        n_layer=2,
        mixer_type="gdn2",
        surprise_net_per_layer=1,  # every layer is GDN-2
        use_short_conv=True,
        expand_v=1.0,
        n_embd=256,
        n_head=8,
        head_size=64,
        intermediate_size=688,
        vocab_size=50257,
    )
    stats = compute_model_params(cfg)
    model = SurpriseGPT(cfg)
    real = sum(p.numel() for p in model.parameters())
    assert stats["total"] == real
    assert stats["num_surprise_layers"] == cfg.n_layer
    assert stats["num_standard_layers"] == 0


def test_when_1_3b_gdn2_config_then_gdn2_core_is_31m() -> None:
    # The reference 1.3B GDN-2 head configuration (18 heads, head_dim=128).
    # Exercises the per-head value dim correction in the analytical total.
    from qwendopamine.models.gdn2.gdn2 import GatedDeltaNet2

    layer = GatedDeltaNet2(
        hidden_size=2304,
        num_heads=18,
        head_dim=128,
        num_v_heads=18,
        expand_v=1.0,
        use_short_conv=True,
        conv_size=4,
        layer_idx=0,
    )
    actual = sum(p.numel() for p in layer.parameters())
    # q/k/v projs + 3 short convs + f/b/w/g projs + o_norm + o_proj + A_log/dt_bias
    k_dim = 18 * 128
    hv = 128
    v_dim = 18 * 128
    expected = (
        (2304 * k_dim * 2)  # q, k
        + (2304 * v_dim)  # v
        + ((k_dim * 4 * 2) + (v_dim * 4))  # short convs
        + ((2304 * hv) + (hv * k_dim))  # f_proj
        + (2304 * k_dim)  # b_proj
        + (2304 * v_dim)  # w_proj
        + ((2304 * hv) + (hv * v_dim) + v_dim)  # g_proj
        + (v_dim * 2304)  # o_proj
        + hv  # o_norm
        + (18 + k_dim)  # A_log (per head) + dt_bias (per key dim)
    )
    assert actual == expected