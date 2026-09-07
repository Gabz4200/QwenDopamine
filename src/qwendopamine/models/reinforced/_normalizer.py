import torch
from torch import Tensor


def normalize_reward_for_advantage(
    reward_values: Tensor,
    running_mean: Tensor | None,
    running_std: Tensor | None,
    *,
    alpha: float = 0.1,
    eps: float = 1e-5,
    training: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Standardise raw reward values to advantage-like scale.

    Implements:

        advantage = (reward - running_mean) / (running_std + eps)

    where ``running_mean`` and ``running_std`` are per-channel EMA
    statistics updated on every call when ``training=True``. Both statistics
    are returned so the caller can persist them in the cache and recover
    them across step-by-step decoding.

    No clip is applied. The standardisation already bounds the signal
    (any outlier is divided by ``running_std``) and the downstream
    ``AdvantageGate`` is itself a ``sigmoid`` so it cannot be driven out
    of bounds by a large advantage. Clipping on top of the standardisation
    silently saturates ``RewardStatisticsExtractor``'s ``max``/``min``
    outputs and kills the gradient on the very values the clip is meant
    to protect. If raw-reward spikes are a concern, feed the layer with
    pre-normalised advantage-like signals (TD error, GAE, surprise) — see
    spec items 6.6 and 8.

    Args:
        reward_values: (B, L, k) raw reward tensor. Any broadcastable shape
            that resolves to (B, L, k) is accepted.
        running_mean: optional (B, k) previous running mean. ``None``
            initialises to zeros; the first batch starts from zero and the
            EMA will quickly adapt.
        running_std: optional (B, k) previous running std (positive). ``None``
            initialises to ones; the EMA is the absolute deviation, so the
            first step will dominate the std estimate.
        alpha: EMA decay in (0, 1]. Larger values weight recent observations
            more heavily. ``0`` disables the EMA update.
        eps: numerical floor for the std division.
        training: when True the running statistics are EMA-updated; when
            False the function is a no-op for the running stats
            (still applies the standardisation with the supplied running
            mean / std).

    Returns:
        ``(normalised, new_mean, new_std)``. ``normalised`` has the same
        shape as ``reward_values``; ``new_mean`` and ``new_std`` are
        (B, k) EMA-updated statistics.
    """
    if reward_values.dim() == 2:
        # (B, k) -> (B, 1, k) so the broadcast below matches the (B, L, k)
        # convention used elsewhere in the layer.
        reward_values = reward_values.unsqueeze(1)
    if reward_values.dim() != 3:
        raise ValueError(
            f"reward_values must be broadcastable to (B, L, k); got shape "
            f"{tuple(reward_values.shape)}."
        )
    B, _, k = reward_values.shape
    device = reward_values.device
    dtype = reward_values.dtype
    if running_mean is None:
        running_mean = torch.zeros(B, k, device=device, dtype=dtype)
    if running_std is None:
        running_std = torch.ones(B, k, device=device, dtype=dtype)

    if training and alpha > 0.0:
        batch_mean = reward_values.mean(dim=1)
        # E[|X - E[X]|^2] to keep the EMA of std cheap and numerically
        # stable under bf16 (a small numerical-floor clamp guards against
        # bf16 producing a slightly negative variance from the subtraction).
        centered_sq = (reward_values - batch_mean.unsqueeze(1)).pow(2)
        batch_var = centered_sq.mean(dim=1).clamp(min=0.0)
        batch_std = batch_var.sqrt()

        new_mean = (1.0 - alpha) * running_mean + alpha * batch_mean
        new_std = (1.0 - alpha) * running_std + alpha * batch_std
    else:
        new_mean = running_mean
        new_std = running_std

    std_safe = new_std.unsqueeze(1) + eps
    normalised = (reward_values - new_mean.unsqueeze(1)) / std_safe
    return normalised, new_mean, new_std
