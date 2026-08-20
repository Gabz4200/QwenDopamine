> **Note:** GatedSurpriseNet was removed from the QwenDopamine codebase. This document is retained for historical reference only.

# Empirical Analysis and Mathematical Boundaries of Precision-Weighted GatedSurpriseNet

## 1. Executive Summary

This document formalizes the mathematical bounds, spectral stability analysis, and empirical findings for the **Precision-Weighted GatedSurpriseNet** module within the `QwenDopamine` framework. 

By replacing the arbitrary independent gate $u_t$ in the original GatedSurpriseNet formulation with a **Precision-Weighted Surprise Metric** ($\pi_t = 1 / \sigma_t^2 \in (0, \infty)^{d_v}$), the recurrent state update is directly driven by local Gaussian negative log-likelihood (NLL) optimization. 

This analysis evaluates the stability boundaries across the write gate ($w_t$), key-erase gate ($b_t$), and precision metric ($\pi_t$), grounded in empirical grid sweeps over 10 real movie loglines from the `razsarusi/plottwist-movies` dataset ($B=10, L=27, D=768$) embedded via GPT-2.

---

## 2. Mathematical Formulation & Spectral Radius Analysis

### 2.1 Closed-Form Recurrent Update
At token step $t$, the memory state matrix $S_t \in \mathbb{R}^{d_k \times d_v}$ optimizes the local precision-weighted regression loss:

$$\min_{S} L_t(S) = \frac{1}{2} \| S - \text{Diag}(\alpha_t) S_{t-1} \|_F^2 + \frac{1}{2} \left( z_t - S^\top (b_t \odot k_t) \right)^\top \text{Diag}(\pi_t) \left( z_t - S^\top (b_t \odot k_t) \right)$$

where:
- $\alpha_t = \exp(g_t) \in (0, 1]^{d_k}$ represents channel-wise memory decay ($g_t \le 0$).
- $b_t \in [0, b_{\text{max}}]^{d_k}$ represents key-side erase gating.
- $w_t \in [0, w_{\text{max}}]^{d_v}$ represents value-side write gating.
- $z_t = w_t \odot v_t$ represents the target write vector.
- $\sigma_t^2 = \text{softplus}(\mathbf{W}_\sigma x_t) + 1e-4$ represents predicted token variance.
- $\pi_t = 1 / \sigma_t^2 \in (0, \pi_{\text{max}}]^{d_v}$ represents the precision multiplier.

Setting $\nabla_S L_t(S) = 0$ yields the state recurrence:

$$S_t = \text{Diag}(\alpha_t) S_{t-1} + (b_t \odot k_t) \left( \pi_t \odot \left( w_t \odot v_t - (\text{Diag}(\alpha_t) S_{t-1})^\top (b_t \odot k_t) \right) \right)^\top$$

### 2.2 Transition Operator Spectrum & Stability Criterion
Let $e_t = b_t \odot k_t \in \mathbb{R}^{d_k}$ with $\|k_t\|_2 = 1.0$. The homogeneous transition operator $\mathcal{M}_j$ acting on value channel $j$ of memory column $S_{:, j} \in \mathbb{R}^{d_k}$ is:

$$\mathcal{M}_j = \left( I - \pi_{t, j} e_t e_t^\top \right) \text{Diag}(\alpha_t)$$

Applying $\mathcal{M}_j$ to eigenvector $e_t$ gives the exact non-trivial eigenvalue $\lambda_j$:

$$\lambda_j = \alpha_{t, \text{eff}} \left( 1 - \pi_{t, j} \|b_t \odot k_t\|_2^2 \right)$$

To prevent exponential state explosion over long sequences ($T \to \infty$), the spectral radius must satisfy $|\lambda_j| \le 1.0$:

$$-1.0 \le \alpha_{t, \text{eff}} \left( 1 - \pi_{t, j} \|b_t \odot k_t\|_2^2 \right) \le 1.0 \implies \pi_{t, j} \le \frac{1 + \frac{1}{\alpha_{t, \text{eff}}}}{\|b_t \odot k_t\|_2^2}$$

This demonstrates mathematically that memory decay ($\alpha_t \approx 0.5$) and key erase norms ($\|b_t \odot k_t\|_2^2 \le 1.0$) expand the stable upper limit for precision $\pi_t$ beyond $1.0$ up to $\pi_{\text{max}} \approx 3.5 - 4.0$ under normal operating conditions.

---

## 3. Empirical Grid Search on Real Token Sequences

To determine non-exploding upper bounds under the asymmetrical constraint $b_{\text{max}} = 2 \cdot w_{\text{max}}$ (matching GDN-2's key-side erase priority), a 3D grid sweep was executed over 10 real movie logline embeddings ($B=10, L=27, D=768$) with active memory decay.

### 3.1 Empirical Results Table ($b_{\text{max}} = 2 \cdot w_{\text{max}}$)

| Write Bound $w_{\text{max}}$ | Erase Bound $b_{\text{max}}$ ($2\times$) | Precision Bound $\pi_{\text{max}}$ | Dynamic Volume ($w \cdot b \cdot \pi$) | Final Output Norm | Final Memory Norm | Empirical Stability Condition |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1.50** | **3.00** | **2.00** | **9.00** | 0.90 | 11.35 | **Well-Stable (Max Balanced)** |
| **1.75** | **3.50** | **1.50** | **9.19** | 1.03 | 13.35 | **Well-Stable (Max Gate Range)** |
| **1.00** | **2.00** | **4.00** | **8.00** | 0.62 | 7.39 | **Well-Stable (Max Precision Range)** |
| **1.25** | **2.50** | **2.50** | **7.81** | 0.60 | 7.12 | **Well-Stable (Ultra-Safe)** |
| 1.25 | 2.50 | 3.00 | 9.38 | 1.22 | 16.14 | Moderate Growth |
| 1.00 | 2.00 | 5.00 | 10.00 | 2.49 | 34.35 | Moderate Growth |
| 2.25 | 4.50 | 1.00 | 10.12 | 2.91 | 40.33 | Moderate Growth |
| 1.50 | 3.00 | 2.50 | 11.25 | 11.79 | 165.38 | High Growth (Diverging) |
| 1.75 | 3.50 | 2.00 | 12.25 | 37.02 | 520.21 | High Growth (Diverging) |
| 1.25 | 2.50 | 4.00 | 12.50 | 48.50 | 681.80 | High Growth (Diverging) |
| 1.50 | 3.00 | $\ge 10.0$ | $\infty$ | $\infty$ | $\infty$ | **EXPLODED** |

### 3.2 Empirical Stability Boundary
Across the empirical evaluation, the hyper-surface separating well-stable execution ($\text{mem\_norm} \le 15.0$) from numerical growth ($\text{mem\_norm} > 100.0 \to \infty$) conforms to:

$$w_{\text{max}}^2 \cdot \pi_{\text{max}} \le 4.50 \iff b_{\text{max}}^2 \cdot \pi_{\text{max}} \le 18.0$$

---

## 4. Theoretical Hypotheses & Open Research Questions

### 4.1 Hypothesis: Precision Gating of the $[0, 2]$ Erase Range
In the GDN-2 paper (arXiv:2605.22791, Section 3.1 & Table 5), the authors reported that expanding the erase gate range from $[0, 1]^{d_k}$ to $[0, 2]^{d_k}$ provided no consistent gain on language modeling benchmarks.

**Working Hypothesis:**
In standard GDN-2, when $b_t > 1.0$, the key-side eigenvalue becomes negative ($\lambda_1 = 1 - b_t < 0$), performing sign-inversion. Without precision weighting, this sign inversion is applied indiscriminately across both informative tokens and ungrounded/noisy tokens, potentially degrading context over long sequences.

Under the Precision-Weighted Surprise Net, $\pi_t$ acts as a precision filter:
- On noisy or uncertain tokens ($\pi_t \to 0$), residual updates are suppressed, mitigating ungrounded sign inversions.
- On high-confidence surprises ($\pi_t \gg 1$), $\pi_t$ amplifies the residual, allowing the $[0, 2]$ erase gate to execute single-step sign inversion rewrites.

*Note: This mechanism is currently an unverified theoretical hypothesis. Full pre-training and downstream benchmark evaluations (e.g. perplexity on FineWeb-Edu, long-context RULER probes) are required to determine whether precision gating converts the $[0, 2]$ erase range expansion into a measurable performance improvement.*

---

## 5. Recommended Configurations for Further Evaluation

Based on the empirical grid search and spectral limits, three candidate configurations are recommended for downstream pre-training experiments:

1. **Option A (Max Precision Range - Surprise Focus):**
   - $w_t \in [0, 1.0]$ (`sigmoid`)
   - $b_t \in [0, 2.0]$ (`2.0 * sigmoid`)
   - $\pi_t \in [0, 4.0]$ (`4.0 * sigmoid`)
   - *Properties:* Maintains standard GDN-2 write and erase bounds while maximizing precision residual amplification.

2. **Option B (Max Balanced Capacity):**
   - $w_t \in [0, 1.50]$ (`1.5 * sigmoid`)
   - $b_t \in [0, 3.00]$ (`3.0 * sigmoid`)
   - $\pi_t \in [0, 2.00]$ (`2.0 * sigmoid`)
   - *Properties:* Expands write and erase capacity by $+50\%$ while maintaining strict spectral stability.

3. **Option C (Max Gate Range):**
   - $w_t \in [0, 1.75]$ (`1.75 * sigmoid`)
   - $b_t \in [0, 3.50]$ (`3.5 * sigmoid`)
   - $\pi_t \in [0, 1.50]$ (`1.5 * sigmoid`)
   - *Properties:* Maximizes key-side erase range for sign-inversion dynamics while capping precision near unit scale.
