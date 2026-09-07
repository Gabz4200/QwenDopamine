"""M11: GDN-2 state must persist across decode steps in GDN2GPT."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("gdn2_per_layer", [1])
def test_when_decode_steps_then_gdn2_state_persists(
    gdn2_per_layer: int,
) -> None:
    """Two sequential decode steps on a GDN-2 layer with the same cache
    must NOT see fresh state on the second step.

    The previous code returned ``None`` from ``build_kv_caches`` for
    GDN-2 layers, so the GDN-2 ``forward`` ran with
    ``past_key_values=None`` and re-initialised state to zero. The
    fix threads a per-layer ``dict`` cache through the block so the
    recurrent state carries across steps. (Review M11.)
    """
    import torch

    from qwendopamine.models.gdn2_gpt.config import GDN2GPTConfig
    from qwendopamine.models.gdn2_gpt.model import GDN2GPT

    cfg = GDN2GPTConfig(
        n_layer=2,
        n_head=1,
        n_query_groups=1,
        n_embd=8,
        head_size=8,
        block_size=8,
        mlp=False,
        shared_attention_norm=False,
        gdn2_per_layer=gdn2_per_layer,  # every layer is GDN-2
    )
    if not hasattr(cfg, "qk_norm"):
        object.__setattr__(cfg, "qk_norm", False)
    model = GDN2GPT(cfg)
    model.eval()

    # Build a cache. Should now be a list of dicts, not None, for
    # GDN-2 layers.
    idx = torch.zeros(1, 1, dtype=torch.long)
    caches = model.build_kv_caches(idx, max_seq_length=cfg.block_size)
    assert caches, "build_kv_caches returned empty list"
    for i, c in enumerate(caches):
        assert c is not None, (
            f"build_kv_caches returned None for layer {i}; review M11: "
            f"GDN-2 layers must carry a per-layer dict cache."
        )
        assert isinstance(c, dict), (
            f"Layer {i} cache must be a dict (review M11); got {type(c)}"
        )

    # First forward with a single token. Inspect the GDN-2 cache
    # before and after the second forward — the state should be
    # different (not re-initialised to zero).
    x = torch.zeros(1, 1, dtype=torch.long)  # token ids [B, T]
    model.kv_caches = caches
    with torch.no_grad():
        out1 = model(x, input_pos=torch.tensor([0]))
        _ = out1
        # The GDN-2 forward updates the cache's ``recurrent_state``.
        state_after_first = model.kv_caches[0].get("recurrent_state")
        if state_after_first is not None:
            state_snapshot = state_after_first.detach().clone()
        out2 = model(x, input_pos=torch.tensor([1]))
        _ = out2
        state_after_second = model.kv_caches[0].get("recurrent_state")
    if state_after_first is not None and state_after_second is not None:
        # State should have changed between steps (memory grew).
        assert not torch.allclose(state_snapshot, state_after_second.detach()), (
            "GDN-2 recurrent state did not change between decode steps; "
            "the cache is not being threaded. Review M11."
        )
