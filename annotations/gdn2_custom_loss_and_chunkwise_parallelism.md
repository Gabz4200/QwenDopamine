# GDN-2 Extensions: Custom Loss Functions and Mathematical Constraints for Chunkwise Parallel Scanning

## 1. Executive Summary

This document formalizes the mathematical and algorithmic constraints for extending **Gated DeltaNet-2 (GDN-2)** and related linear recurrent memory architectures within the `QwenDopamine` framework.

While standard linear attention networks allow arbitrary sequential state updates during token-by-token inference ($O(T)$ complexity), scaling these models to large context windows and multi-billion parameter sizes requires **chunkwise parallel training** ($O(\log T)$ parallel scan complexity).

This analysis grounds the update rules of GDN-2 (arXiv:2605.22791) in two fundamental conditions required to derive the **WY representation** and intra-chunk triangular solve system. Any custom memory loss, gating mechanism, or rank expansion introduced in `src/qwendopamine/models/blocks/` must conform strictly to these two principles to maintain parallel GPU training capabilities.

---

## 2. Theoretical Grounding in GDN-2

In GDN-2, the recurrent update of the key-value memory state matrix $S_t \in \mathbb{R}^{d_k \times d_v}$ is derived as the analytical solution to an online local regression problem at step $t$:

$$\min_{S} L_t(S) = \frac{1}{2} \| S - \text{Diag}(g_t) S_{t-1} \|_F^2 - \langle S k_t, \, w_t \odot v_t - (\text{Diag}(g_t) S_{t-1})^\top (b_t \odot k_t) \rangle$$

Setting the gradient $\nabla_S L_t(S) = 0$ yields the GDN-2 state transition update:

$$S_t = \underbrace{\left( I - k_t (b_t \odot k_t)^\top \right) \text{Diag}(g_t)}_{\text{Transition Operator } M_t} S_{t-1} + \underbrace{k_t (w_t \odot v_t)^\top}_{\text{Input Update } N_t}$$

where:
- $g_t = \exp(-\gamma_t) \in (0, 1]^{d_k}$ represents the channel-wise decay gate.
- $b_t \in [0, 1]^{d_k}$ represents the channel-wise erase gate on keys.
- $w_t \in [0, 1]^{d_v}$ represents the channel-wise write gate on values.

The chunkwise parallel scan algorithm converts sequence updates across a block of size $C$ into parallel matrix operations by factoring the state transitions into a normalized recurrence.

---

## 3. Condition 1: Affine Linearity of the State Update

### Mathematical Requirement
The local loss function $L_t(S)$ must be **quadratic** with respect to $S$, or the state update $S_t = f(S_{t-1}, x_t)$ must be an **affine mapping** (linear transformation plus translation) with respect to the prior memory state $S_{t-1}$:

$$S_t = M_t S_{t-1} + N_t$$

where $M_t \in \mathbb{R}^{d_k \times d_k}$ and $N_t \in \mathbb{R}^{d_k \times d_v}$ are independent of $S_{t-1}$.

### Derivation & Parallel Proof
In a chunk of length $C$, the cumulative state recurrence can be expanded inductively:

$$\bar{S}_C = \left[ \prod_{r=1}^C M_r \right] \bar{S}_0 + \sum_{r=1}^C \left[ \prod_{s=r+1}^C M_s \right] N_r$$

Because the operator $M_r$ is linear with respect to $\bar{S}_{r-1}$, the initial state $\bar{S}_0$ factors out cleanly from the intra-chunk sequence inputs $(\bar{K}, \bar{E}, \bar{Z})$.

### Failure Modes of Non-Quadratic Loss Functions
If the local objective $L_t(S)$ is replaced with a non-quadratic function (such as $L_1$ norm, Cosine Distance, Huber Loss, or Cross-Entropy):

1. **Non-Linear Gradient Terms:** $\nabla_S L_t(S)$ introduces non-linear dependencies on $S_{t-1}$ (e.g., $\text{sign}(S_{t-1}^\top e_t)$ or $\text{softmax}(S_{t-1}^\top e_t)$).
2. **Loss of Factorization:** The initial state $\bar{S}_0$ becomes nested inside activation functions during time step unrolling.
3. **Computational Bottleneck:** The initial state cannot be separated from intra-chunk inputs, forcing execution to fall back to sequential token-by-token evaluation ($O(T)$), which is prohibitively slow for training.

---

## 4. Condition 2: Low-Rank Perturbation Structure of $M_t - I$

### Mathematical Requirement
The state transition operator $M_t$ acting on $S_{t-1}$ must be a **low-rank perturbation of the identity matrix** (or of a diagonal decay matrix).

For GDN-2, the perturbation is rank-1:

$$M_t - I = - \bar{k}_t \bar{e}_t^\top$$

For multi-rank variants, the perturbation must take the form of a rank-$r$ outer product sum where $r \ll d_k$:

$$M_t - I = - \sum_{i=1}^r \bar{k}_{i, t} \bar{e}_{i, t}^\top$$

### Derivation & The WY System Solve
The chunkwise WY representation contracts the product of $C$ rank-1 updates into a compact matrix inverse system:

$$\prod_{r=1}^C (I - \bar{k}_r \bar{e}_r^\top) = I - \bar{K}^\top A \bar{E}$$

where:
- $\bar{K} = [\bar{k}_1, \dots, \bar{k}_C]^\top \in \mathbb{R}^{C \times d_k}$
- $\bar{E} = [\bar{e}_1, \dots, \bar{e}_C]^\top \in \mathbb{R}^{C \times d_k}$
- $T = \text{tril}(\bar{E} \bar{K}^\top, -1) \in \mathbb{R}^{C \times C}$ is a strictly lower-triangular matrix.
- $A = (I + T)^{-1} \in \mathbb{R}^{C \times C}$ is a unit lower-triangular matrix solved via fast forward substitution.

Using this formulation, the state at the end of the chunk is computed in parallel:

$$S_C = \text{Diag}(\gamma_C) S_0 + \bar{K}^\top \underbrace{A (Z - \bar{E} S_0)}_{R \in \mathbb{R}^{C \times d_v}}$$

### Failure Modes of Dense Perturbations
If $M_t - I$ is a full-rank $d_k \times d_k$ matrix (such as an arbitrary multi-layer perceptron transformation acting directly on state $S_{t-1}$):

1. **Dimensionality Expansion:** The interaction matrix $T$ can no longer be constructed as a $C \times C$ lower-triangular matrix.
2. **Computational Overhead:** Evaluating the intra-chunk state transition requires multiplying full $d_k \times d_k$ dense matrices at every step, incurring $O(C \cdot d_k^3)$ complexity instead of the efficient $O(C^2 d_k)$ intra-chunk matrix multiplication.

---

## 5. Architectural Taxonomy of GDN-2 Modifications

The following table summarizes which architectural modifications are compatible with chunkwise parallel training under these two conditions:

| Proposed Architecture / Modification | Parallel Scan Compatible? | Governing Principle & Mathematical Explanation |
| :--- | :---: | :--- |
| **Rank-$r$ Delta Rule Expansion** | **YES** | Replaces single key-erase outer product with $\sum_{i=1}^r k_{i,t} e_{i,t}^\top$. Expands $T$ to $(rC) \times (rC)$, remaining efficient for small $r$ (e.g. $r \in \{2, 4\}$). |
| **Weighted $L_2$ Loss / Mahalanobis Metric** | **YES** | Local loss $L_t = \frac{1}{2} \| S^\top e_t - z_t \|_W^2$ preserves affine gradient linearity in $S_{t-1}$. |
| **Channel-wise Independent Decays** | **YES** | Decays $\text{Diag}(g_t)$ are absorbed into normalized key/erase tensors $\bar{k}_r, \bar{e}_r$ without breaking linearity. |
| **Multi-Head State Decomposition** | **YES** | Independent per-head state matrices $S_h \in \mathbb{R}^{d_k \times d_v}$ maintain independent parallel WY solves per head. |
| **Non-Linear State Transitions (e.g. $\text{SiLU}(S_t)$)** | **NO** | Violates Condition 1. Non-linear functions prevent factoring $S_0$ out of sequence products. |
| **Softmax / Cosine Local Loss** | **NO** | Violates Condition 1. Gradient depends non-linearly on $S_{t-1}$, breaking associative grouping. |
| **Dense State Multi-Layer Perceptron (State MLP)** | **NO** | Violates Condition 2. Full-rank matrix updates prevent the $C \times C$ WY triangular solve. |

---

## 6. Engineering Implications for `QwenDopamine`

When implementing or extending custom transformer blocks in `src/qwendopamine/models/blocks/registry.py`:

1. **Sequential vs. Parallel Implementations:**
   Experimental blocks utilizing non-quadratic local losses may be registered for CPU-based inference and probing using token-by-token loops. However, they must be flagged as non-parallelizable and excluded from standard chunkwise GPU pre-training pipelines.

2. **Validation of Custom Block Proposals:**
   Before adding a new linear recurrence block to `BLOCKS`, verify that its state update can be expressed as:
   $$S_t = \left( \text{Diag}(g_t) - \sum_{i=1}^r k_{i,t} e_{i,t}^\top \right) S_{t-1} + N_t$$
   This ensures compatibility with high-throughput Triton GPU kernels and chunkwise parallel scans.
