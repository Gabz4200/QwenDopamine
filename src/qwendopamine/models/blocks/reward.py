r"""Reward conditioning encoder and spatial positional encoding blocks."""

import math
from typing import cast

import torch
from einops import rearrange
from torch import nn


class LearnableFourierFeatures(nn.Module):
    r"""LearnableFourierFeatures(pos_dim, f_dim, h_dim, d_dim, g_dim=1, gamma=1.0, include_input=True)

    Applies learnable Fourier feature mapping and MLP projection for multi-dimensional spatial
    positional encodings.

    .. math::
        F = \frac{[\cos(x W_r^T), \sin(x W_r^T)]}{\sqrt{f\_dim}}

    Args:
        pos_dim (int): Dimension of input position coordinate vector.
        f_dim (int): Number of Fourier feature channels (must be divisible by 2).
        h_dim (int): Hidden dimension of the internal MLP projection network.
        d_dim (int): Final output feature dimension (must be divisible by ``g_dim``).
        g_dim (int, optional): Group dimension factor for output feature reshaping. Default: ``1``.
        gamma (float, optional): Scaling hyperparameter for initial random projection matrix weights.
            Default: ``1.0``.
        include_input (bool, optional): If ``True``, concatenates raw position vector to Fourier features
            before MLP projection. Default: ``True``.

    Examples::

        >>> lff = LearnableFourierFeatures(pos_dim=4, f_dim=16, h_dim=32, d_dim=64)
        >>> pos = torch.randn(2, 5, 1, 4)
        >>> out = lff(pos)
        >>> out.shape
        torch.Size([2, 5, 64])
    """

    def __init__(
        self,
        pos_dim: int,
        f_dim: int,
        h_dim: int,
        d_dim: int,
        g_dim: int = 1,
        gamma: float = 1.0,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        assert f_dim % 2 == 0, (
            "number of fourier feature dimensions must be divisible by 2."
        )
        assert d_dim % g_dim == 0, (
            "number of D dimension must be divisible by the number of G dimension."
        )
        self.include_input = include_input
        self.div_term = math.sqrt(f_dim)
        enc_f_dim = int(f_dim / 2)
        dg_dim = int(d_dim / g_dim)

        self.Wr = nn.Parameter(torch.randn([enc_f_dim, pos_dim]) * (gamma**2))

        mlp_in_dim = f_dim + pos_dim if include_input else f_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, h_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(h_dim, dg_dim),
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        r"""forward(pos) -> Tensor

        Args:
            pos (Tensor): Input position coordinates of shape :math:`(B, L, G, \text{pos\_dim})`.

        Returns:
            Tensor: Encoded spatial positional features of shape :math:`(B, L, \text{d\_dim})`.
        """
        XWr = torch.matmul(pos, self.Wr.T)
        F = torch.cat([torch.cos(XWr), torch.sin(XWr)], dim=-1) / self.div_term

        if self.include_input:
            F = torch.cat([pos, F], dim=-1)

        Y = self.mlp(F)
        pos_enc = rearrange(Y, "b l g d -> b l (g d)")
        return pos_enc


class FourierFeatures(nn.Module):
    r"""FourierFeatures(pos_dim, f_dim, sigma=10.0, train=False, include_input=True)

    Applies random Fourier feature encoding to map low-dimensional positions to high-frequency
    representations.

    .. math::
        F = [\sin(x B), \cos(x B)]

    Args:
        pos_dim (int): Dimension of input position coordinate vector.
        f_dim (int): Number of Fourier feature channels (must be divisible by 2).
        sigma (float, optional): Standard deviation scale for initial Gaussian projection matrix.
            Default: ``10.0``.
        train (bool, optional): If ``True``, registers projection matrix as a trainable parameter.
            Default: ``False``.
        include_input (bool, optional): If ``True``, concatenates raw position vector to Fourier features.
            Default: ``True``.

    Examples::

        >>> ff = FourierFeatures(pos_dim=2, f_dim=16)
        >>> pos = torch.randn(2, 10, 2)
        >>> enc = ff(pos)
        >>> enc.shape
        torch.Size([2, 10, 18])
    """

    def __init__(
        self,
        pos_dim: int,
        f_dim: int,
        sigma: float = 10.0,
        train: bool = False,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        assert f_dim % 2 == 0, "number of channels must be divisible by 2."
        self.include_input = include_input
        enc_dim = int(f_dim / 2)

        B = torch.randn([pos_dim, enc_dim]) * sigma
        if train:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        r"""forward(pos) -> Tensor

        Args:
            pos (Tensor): Input position tensor of shape :math:`(..., \text{pos\_dim})`.

        Returns:
            Tensor: Encoded position features of shape :math:`(..., \text{f\_dim} + \text{pos\_dim})`
                if ``include_input=True``, else :math:`(..., \text{f\_dim})`.
        """
        proj = torch.matmul(pos, self.B)
        pos_enc = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        if self.include_input:
            pos_enc = torch.cat([pos, pos_enc], dim=-1)

        return pos_enc


class PositionalEncoding(nn.Module):
    r"""PositionalEncoding(pos_dim, enc_dim, include_input=True)

    Computes fixed sinusoidal positional encodings using geometric frequency scaling.

    .. math::
        \text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad
        \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)

    Args:
        pos_dim (int): Spatial or coordinate dimension of input position.
        enc_dim (int): Encoding dimension (must be even).
        include_input (bool, optional): If ``True``, concatenates raw position input to encodings.
            Default: ``True``.

    Examples::

        >>> pe = PositionalEncoding(pos_dim=1, enc_dim=16)
        >>> pos = torch.randn(2, 8, 1)
        >>> out = pe(pos)
        >>> out.shape
        torch.Size([2, 8, 17])
    """

    def __init__(
        self,
        pos_dim: int,
        enc_dim: int,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        assert enc_dim % 2 == 0, "dimension of positional encoding must be even."
        self.include_input = include_input
        half_enc_dim = int(enc_dim / 2)
        div_term = torch.exp(
            torch.arange(0, half_enc_dim) * -(math.log(10000.0) / half_enc_dim)
        )
        freqs = div_term.unsqueeze(0).expand(pos_dim, -1)

        self.register_buffer("freqs", freqs)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        r"""forward(pos) -> Tensor

        Args:
            pos (Tensor): Input position tensor of shape :math:`(..., \text{pos\_dim})`.

        Returns:
            Tensor: Positional encoding tensor of shape :math:`(..., \text{enc\_dim} + \text{pos\_dim})`
                if ``include_input=True``, else :math:`(..., \text{enc\_dim})`.
        """
        freqs = cast(torch.Tensor, self.freqs)
        proj = torch.matmul(pos, freqs)
        pos_enc = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        if self.include_input:
            pos_enc = torch.cat([pos, pos_enc], dim=-1)

        return pos_enc


class AdaLN(nn.Module):
    r"""AdaLN(dim, eps=1e-6)

    Adaptive Layer Normalization modulating normalized features using externally supplied scale (:math:`\gamma`)
    and shift (:math:`\beta`) conditioning parameters.

    .. math::
        \text{AdaLN}(x, [\gamma, \beta]) = \text{RMSNorm}(x) \odot \gamma + \beta

    Args:
        dim (int): Feature dimension of normalized input tensor.
        eps (float, optional): Small epsilon constant for RMSNorm stability. Default: ``1e-6``.

    Examples::

        >>> adaln = AdaLN(dim=32)
        >>> x = torch.randn(2, 5, 32)
        >>> cond = torch.randn(2, 5, 64)
        >>> out = adaln(x, cond)
        >>> out.shape
        torch.Size([2, 5, 32])
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.norm = nn.RMSNorm(dim, eps=eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        r"""forward(x, cond) -> Tensor

        Args:
            x (Tensor): Input tensor of shape :math:`(B, L, D)` or :math:`(B, D)`.
            cond (Tensor): Conditioning tensor containing concatenated scale and shift parameters of
                shape :math:`(B, 2D)` or :math:`(B, L, 2D)`.

        Returns:
            Tensor: Adaptively normalized and modulated feature tensor of same shape as ``x``.
        """
        if cond.dim() == 2 and x.dim() == 3:
            cond = cond.unsqueeze(1)

        gamma, beta = cond.chunk(2, dim=-1)
        x_norm = self.norm(x)
        return x_norm * gamma + beta


class RewardEncoder(nn.Module):
    r"""RewardEncoder(dim, hidden_dim, eps=1e-6, f_dim=32, h_dim=64, gamma=1.0)

    Encodes sequences of unbounded reward signals into statistical summary vectors (mean, median, max, min),
    projects them via learnable Fourier features, and adaptively modulates input embedding sequences using :class:`AdaLN`.

    .. math::
        s = [\text{median}(R), \text{mean}(R), \text{max}(R), \text{min}(R)] \in \mathbb{R}^{B \times L \times 4}

    .. math::
        [\gamma, \beta] = \text{LearnableFourierFeatures}(s), \quad
        y = \text{AdaLN}(W_{\text{proj}} x, [\gamma, \beta])

    Args:
        dim (int): Input sequence feature dimension.
        hidden_dim (int): Target hidden feature dimension.
        eps (float, optional): RMSNorm stability epsilon for internal :class:`AdaLN`. Default: ``1e-6``.
        f_dim (int, optional): Number of Fourier channels for reward statistical encoding. Default: ``32``.
        h_dim (int, optional): Hidden dimension of Fourier MLP projection network. Default: ``64``.
        gamma (float, optional): Scaling hyperparameter for initial Fourier projection weights. Default: ``1.0``.

    Examples::

        >>> encoder = RewardEncoder(dim=32, hidden_dim=64)
        >>> x = torch.randn(2, 5, 32)
        >>> reward_values = torch.randn(2, 5, 10)  # 10 different reward loss functions
        >>> output = encoder(x, reward_values)
        >>> output.shape
        torch.Size([2, 5, 64])
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        eps: float = 1e-6,
        f_dim: int = 32,
        h_dim: int = 64,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        if dim != hidden_dim:
            self.x_proj: nn.Module = nn.Linear(dim, hidden_dim)
        else:
            self.x_proj = nn.Identity()

        self.reward_fourier = LearnableFourierFeatures(
            pos_dim=4,
            f_dim=f_dim,
            h_dim=h_dim,
            d_dim=2 * hidden_dim,
            g_dim=1,
            gamma=gamma,
            include_input=True,
        )

        self.adaln = AdaLN(dim=hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor, reward_values: torch.Tensor) -> torch.Tensor:
        r"""forward(x, reward_values) -> Tensor

        Args:
            x (Tensor): Input sequence feature tensor of shape :math:`(B, L, \text{dim})`, :math:`(B, \text{dim})`,
                or :math:`(\text{dim},)`.
            reward_values (Tensor): Reward tensor of shape :math:`(B, L, K)` for :math:`K` unbounded reward
                signals, :math:`(B, L)` for a single reward signal, :math:`(B, K)` for single-step rewards,
                or :math:`(K,)` / :math:`(L,)` / :math:`(L, K)` for unbatched signals.

        Returns:
            Tensor: Modulated sequence feature tensor matching the input shape structure of ``x`` with feature
                dimension :math:`\text{hidden\_dim}`.
        """
        if reward_values.device != x.device or reward_values.dtype != x.dtype:
            reward_values = reward_values.to(device=x.device, dtype=x.dtype)

        orig_x_dim = x.dim()
        if orig_x_dim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif orig_x_dim == 2:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape

        if reward_values.dim() == 1:
            if reward_values.shape[0] == seq_len:
                reward_values = reward_values.view(1, seq_len, 1)
            else:
                reward_values = reward_values.view(batch_size, seq_len, -1)
        elif reward_values.dim() == 2:
            if reward_values.shape == (batch_size, seq_len):
                reward_values = reward_values.unsqueeze(-1)
            elif seq_len == 1 and reward_values.shape[0] == batch_size:
                reward_values = reward_values.unsqueeze(1)
            elif batch_size == 1 and reward_values.shape[0] == seq_len:
                reward_values = reward_values.unsqueeze(0)
            else:
                reward_values = reward_values.unsqueeze(-1)

        reward_median = reward_values.median(dim=-1, keepdim=True)[0]
        reward_mean = reward_values.mean(dim=-1, keepdim=True)
        reward_max = reward_values.max(dim=-1, keepdim=True)[0]
        reward_min = reward_values.min(dim=-1, keepdim=True)[0]

        reward_stats = torch.cat(
            [reward_median, reward_mean, reward_max, reward_min], dim=-1
        )

        reward_stats_reshaped = reward_stats.unsqueeze(2)
        cond = self.reward_fourier(reward_stats_reshaped)
        x_hidden = self.x_proj(x)
        output = self.adaln(x_hidden, cond)

        if orig_x_dim == 1:
            output = output.squeeze(0).squeeze(0)
        elif orig_x_dim == 2:
            output = output.squeeze(1)

        return output
