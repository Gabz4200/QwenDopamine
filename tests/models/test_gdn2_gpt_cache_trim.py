"""M12: GDN2GPT KV cache must be trimmed to max_seq_length."""

from __future__ import annotations


def test_when_decode_steps_exceed_max_seq_length_then_cache_capped() -> None:
    """After multiple decode steps, the cache seq-len never exceeds max_seq_length."""
    from qwendopamine.models.gdn2_gpt.attention import CausalSelfAttention
    from qwendopamine.models.gdn2_gpt.config import GDN2GPTConfig

    cfg = GDN2GPTConfig(
        n_layer=1,
        n_head=1,
        n_query_groups=1,
        n_embd=8,
        head_size=8,
        block_size=4,
        mlp=True,
        shared_attention_norm=True,
    )
    # Force-disable qk_norm (the GDN2GPTConfig default may not expose it).
    if not hasattr(cfg, "qk_norm"):
        object.__setattr__(cfg, "qk_norm", False)
    attn = CausalSelfAttention(cfg, layer_idx=0, n_embd=cfg.n_embd)
    max_seq_length = 4
    # Step 1: seed with an empty cache.
    import torch

    x = torch.randn(1, 1, cfg.n_embd)
    empty_k = torch.zeros(1, 1, 0, cfg.head_size)
    empty_v = torch.zeros(1, 1, 0, cfg.head_size)
    _, cache = attn(
        x, rope=None, max_seq_length=max_seq_length, kv_cache=(empty_k, empty_v)
    )
    assert cache is not None
    cache_k, _cache_v = cache
    assert cache_k.shape[2] == 1
    # Step 2: another single token. Cache now has 2 tokens.
    _, cache = attn(x, rope=None, max_seq_length=max_seq_length, kv_cache=cache)
    cache_k, _cache_v = cache
    assert cache_k.shape[2] == 2
    # Step 3: another. Cache now has 3 tokens.
    _, cache = attn(x, rope=None, max_seq_length=max_seq_length, kv_cache=cache)
    cache_k, _cache_v = cache
    assert cache_k.shape[2] == 3
    # Step 4: another. Cache should be at the cap (4).
    _, cache = attn(x, rope=None, max_seq_length=max_seq_length, kv_cache=cache)
    cache_k, _cache_v = cache
    assert cache_k.shape[2] == max_seq_length
    # Step 5+: must NOT grow past the cap.
    for _ in range(5):
        _, cache = attn(x, rope=None, max_seq_length=max_seq_length, kv_cache=cache)
        cache_k, _cache_v = cache
        assert cache_k.shape[2] <= max_seq_length, (
            f"Cache exceeded max_seq_length={max_seq_length}: got shape {cache_k.shape}"
        )
