"""M14: norm_2 allocation must be consistent with its use-site."""

from __future__ import annotations


def test_when_block_with_shared_attn_norm_and_mlp_then_no_attribute_error() -> None:
    """Constructing a Block with shared_attention_norm=True, mlp=True,
    parallel_residual=False must not crash on forward.

    The previous code only allocated ``norm_2`` when
    ``not shared_attention_norm and not parallel_residual`` was true.
    The forward path uses ``self.norm_2`` whenever
    ``mlp and not parallel_residual``, so the combination
    ``shared_attention_norm=True, parallel_residual=False`` triggered
    AttributeError. Review M14: norm_2 is now always allocated when
    MLP is present.
    """
    from qwendopamine.models.gdn2_gpt.block import Block
    from qwendopamine.models.gdn2_gpt.config import GDN2GPTConfig

    cfg = GDN2GPTConfig(
        n_layer=1,
        n_head=1,
        n_embd=8,
        block_size=16,
        mlp=True,
        shared_attention_norm=True,
        parallel_residual=False,
    )
    block = Block(cfg, layer_idx=0)
    assert hasattr(block, "norm_2"), (
        "Block must always allocate norm_2 when MLP is present (review M14)."
    )
