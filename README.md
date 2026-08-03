# rank-decomposition-lora
The full math of LoRA, measured on real SLM weights — SVD to rank selection to scale-out (135M→1.7B) to deploy-time merging. CPU-only, no assumptions.

---

## Units

| Unit | Topic | Core formula | Status |
|------|-------|-------------|--------|
| 1 | SVD & Eckart–Young on real weights | `W ≈ Uₖ·Sₖ·Vₖᵀ`, error ∝ tail energy | ✅ Complete |
| 2 | The LoRA parametrization | `ΔW = (α/r)·B·A`, budget, merge trick | ✅ Complete |
| 3 | The LoRA hypothesis (real GD) | learned `ΔW` after gradient descent on one layer | 🚧 scaffolded |
| 4 | Rank selection | how to pick `r` with data | pending |
| 5 | Scale-out: spectra 135M→1.7B | rank behavior across model size | pending |
| 6 | Industrial deployment | merging, QLoRA, multi-adapter, CPU serving | pending |

---

## Repository Structure

```
rank-decomposition-lora/
├── README.md                     ← this file
├── requirements.txt              ← pinned env (torch 2.13.0, transformers 5.14.1)
├── venv/                         ← local environment
├── unit1_svd_eckart_young.ipynb  ← low rank measured & exactly quantified
├── unit2_lora_math.ipynb         ← parametrization, budget, merge trick
├── unit3_lora_hypothesis.ipynb  ← GD LoRA vs full FT: does the optimizer find low-rank?
├── unit4_rank_selection.ipynb    ← (scaffold)
├── unit5_scaleout_spectra.ipynb  ← (scaffold) 135M→1.7B, ~4GB downloads
└── unit6_industrial_deployment.ipynb ← (scaffold)
```

Each notebook is **self-contained** (loads the model fresh), runs on CPU-only Windows, and follows the *hypothesis → measurement → verdict* narrative.

---

## Unit 1 — SVD & Eckart–Young on real SmolLM2-135M weights (complete)

**Low rank is a measured, exactly-quantified property** — not a slogan. CB 1.1 SVD'd the five key matrices; CB 1.2 verified the Eckart–Young theorem (|meas−theory| ≤ 9.5e-06); CB 1.3 carried the error through a real forward step (output rel_err 0.2867 vs matrix 0.3148).

The numbers are read comparatively: laying all five rows side by side answers "which matrices are concentrated, which are flat?" And the measurement just caught the most interesting contrast in the whole unit:

- The FFN trio clusters together — k90/n ≈ 0.67-0.69, top1 ≈ 0.6-3%. Flat, spread-out spectra. This reproduces repo 2 (W_gate 394/576 ✅).
- W_E is sharply concentrated — top1 = 50.6%, matching repo 1 exactly (✅). One direction holds half the embedding's energy (the "common" direction shared by all tokens).
- W_Q is the surprise — k@90 = 65/576 → k90/n = 0.113, top1 = 25.4%. The query projection is 6× more concentrated than the FFN matrices. Nobody predicted that; the "~0.7 everywhere" assumption is dead. This is the "measured not assumed" principle paying off.

That W_Q result matters for LoRA: it hints the attention matrices may tolerate much smaller rank than FFN matrices — a hypothesis Unit 4 will test directly.

### Where tokens DO enter (to keep the distinction sharp)

- **Weight-side (this unit):** static spectra — what we just measured.
- **Token-side (repo 2's Unit 3):** the activation rank — A = [silu(W_gate x_t) ⊙ (W_up x_t)] built from real tokens, giving 45/1536 ever-strong dims. That was input-dependent.
- **Bridge (repo 3, CB 1.3):** the trust check uses x0 = X[0] — the first place a token touches this unit's numbers, and it only uses the weight spectrum, not changes it.

### The real lesson: the tail is long

Errors shrink slowly. This matrix's singular values don't die off quickly — they trail off. So W_gate is **approximately low-rank, not strictly low-rank**. And that's precisely why LoRA doesn't try to reproduce W with a small matrix — it tries to perturb it with a small one (W_eff = W₀ + ΔW). You don't replace the model's structure; you nudge a few directions of it. The curve is the empirical justification for that design.

One contrast for later: W_Q (k@90 = 65) would hit error 0.316 with far fewer directions than W_gate's 394. Same formula, different curve shape. The theorem is universal; each matrix just wears it differently.

---

## Unit 2 — The LoRA parametrization (complete)

**Adapting is a budget question, measured exactly.** CB 2.1 computed adapter parameter counts on all five real shapes; CB 2.2 built `ΔW = (α/r)·B·A` and proved the rank bound numerically; CB 2.3 verified the merge trick to float noise.

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| 1 | LoRA params are a small fraction of W | ~2% at r=8 | **0.48%–2.78%** by shape (W_Q 2.78%, FFN 1.91%, W_E 1.41%) | ✅ Holds |
| 2 | `ΔW` is exactly rank ≤ r | ≤ 8 by construction | **8** nonzero singular values = r | ✅ Holds |
| 3 | `ΔW` is a small perturbation of W | `‖ΔW‖/‖W‖` ~ 1e-2..1e-3 | **6.1e-3** | ✅ Holds (in band) |
| 4 | Merged == unmerged | max err ~ 1e-6 | **2.19e-07** (float noise) | ✅ Holds |

**The budget formula:** `fraction = r(d+k)/(dk) = r/d + r/k`. Three consequences the table shows:
- **Linear in r** — r=8 → 1.91% but r=64 → 15.28% on W_gate; rank is a linear budget, not a curve. This is the concrete reason Unit 4 (rank selection) exists.
- **Shape-dependent, not size-dependent** — W_E (49,152×576) costs only 1.41% because its fraction ≈ `r/k`; the *small* dimension sets the price.
- **Symmetric in (d,k)** — W_down (576,1536) costs exactly what W_gate (1536,576) costs: 16,896 params.

**The two serving strategies are the same math:** merged (`W_eff = W₀ + (α/r)BA`, one matmul, zero overhead — deployment default) vs unmerged (`W₀x + (α/r)·B(Ax)`, two small matmuls, base weight shared — multi-adapter routing). Verified identical to 2.19e-07.

**Correction the measurement caught:** the scaffold predicted W_E at ~0.03% — that was wrong; the correct value is **~1.41%** (`r/k = 8/576`). The prediction-as-typo was fixed; the real measured row stands on its own.

**Unit 2 in two lines:** LoRA is Unit 1's spectrum turned into a budget — a rank-8 `(α/r)·B·A` costs ~2% of the parameters, is confined to exactly 8 directions by construction, moves W by ~0.6%, and merges for zero inference overhead.

**Next — Unit 3:** the hypothesis itself — train a real LoRA (r=8) on one layer with gradient descent and SVD the *learned* `ΔW`. Does the optimizer actually find a delta that uses its 8 directions well?

---

## Unit 3 — The LoRA hypothesis (scaffolded)
