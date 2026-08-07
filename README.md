# rank-decomposition-lora
The full math of LoRA, measured on real SLM weights — SVD to rank selection to scale-out (135M→1.7B) to deploy-time merging. CPU-only, no assumptions.

---

## Units

| Unit | Topic | Core formula | Status |
|------|-------|-------------|--------|
| 1 | SVD & Eckart–Young on real weights | `W ≈ Uₖ·Sₖ·Vₖᵀ`, error ∝ tail energy | ✅ Complete |
| 2 | The LoRA parametrization | `ΔW = (α/r)·B·A`, budget, merge trick | ✅ Complete |
| 3 | The LoRA hypothesis (real GD) | learned `ΔW` after gradient descent on one layer | 🚧 CB 3.2 re-run pending |
| 4 | Rank selection | how to pick `r` with data | ✅ Complete (Exp A–C) |
| 5 | Scale-out: spectra 135M→1.7B | rank behavior across model size | pending |
| 6 | Industrial deployment | merging, QLoRA, multi-adapter, CPU serving | pending |

---

## Repository Structure

```
rank-decomposition-lora/
├── README.md                     ← this file
├── requirements.txt              ← pinned env (torch 2.13.0, transformers 5.14.1)
├── .venv/                        ← local environment (hidden, gitignored)
├── unit1_svd_eckart_young.ipynb  ← low rank measured & exactly quantified
├── unit2_lora_math.ipynb         ← parametrization, budget, merge trick
├── unit3_lora_hypothesis.ipynb   ← GD LoRA vs full FT: does the optimizer find low-rank?
├── unit4_rank_selection.ipynb   ← r-sweep measured: loss-elbow on controlled + real deltas
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

## Unit 3 — The LoRA hypothesis: does GD find a low-rank delta? (in progress)

**Design.** A *controlled* task: inject a known rank-4 delta `E` (`‖E‖/‖W‖ = 0.1`) into `W_gate`, train to recover it from 576 real token embeddings (`X = W_E[:576]`, measured `rank(X) = 565`). Then train full fine-tuning on the identical task. The controlled target makes any residual an *optimization* story, not a *representation* one.

**Measured findings (SmolLM2-135M, GD on real tokens — not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| 1 | r=8 LoRA learns the rank-4 delta | rel. loss ~1e-4 or below | **0.0086 → ~5e-6** (3 orders) | ✅ Holds |
| 2 | Learned `ΔW` is concentrated | ~4 strong svals, ~4 tail | ⏳ pending clean re-run | — |
| 3 | Learned subspace aligns with target | overlap ~1.0 | ⏳ pending clean re-run | — |
| 4 | Full FT: more params, no new capability | loss ≥ LoRA, 52.36× params | **slower + ~25% worse floor** (1.1e-5 vs ~5e-6) | ✅ Holds — *prediction corrected* |

**The prediction correction (row 4).** Full fine-tuning was expected to drop at least as far (no representation limit). Measured: at the same 300-step/Adam budget it converged ~2× slower and plateaued ~25% higher, with **52.36× the parameters**. The `r=8` constraint is an inductive bias that already matches the task's true shape; full FT must rediscover low rank inside 884,736 dims, and its extra capacity adds noise, not signal, at equal budget. Honest caveat: both floors are optimizer-limited (Adam lr 1e-2, no decay); the claim is about the budget regime, not the asymptotic limit.

**Process finding — the one-character bug.** A typo `dW = scale = (B @ A)` re-assigned the scalar `scale` (α/r = 2) to a matrix, making the extracted `dW_learned` an elementwise square of rank-8 — producing `matrix_rank = 13`, a smeared spectrum, overlap ~0.1, and a grad-graph plot error, all while the in-loop training stayed valid. The rank gate in CB 3.2 is exactly what caught it. Lesson recorded: the measuring instrument needs gates too.

**Rows 2–3** await the clean re-run of CB 3.1 + 3.2 (fix already applied in the notebook): expected `matrix_rank ≤ 8`, `strong ≈ 4`, `overlap → [~1.0 ×4]`.

---

## Unit 4 — Choosing the Rank: when does adding r stop buying? (complete)

**Design.** Sweep rank `r` across two deltas: a controlled rank-4 delta on `W_gate` (ground truth known) and a *real* cross-layer delta `W_Q1 − W_Q0`. Read the **loss-elbow** (smallest `r` where the next rank buys < 2×) and cross-check it against the delta's spectral effective rank (`k90`). All on 576 real token embeddings (`rank(X) = 565`, quantified caveat).

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| 1 | Controlled sweep: elbow at true rank | r = 4 | **elbow = 4**, floor 3.0e-3 → 2.9e-6 | ✅ Holds |
| 2 | r=2 under-fits by orders | floor ~1e-2..1e-3 | **3.2e-3 vs 2.9e-6** (3 orders) | ✅ Holds |
| 3 | Real Q-delta elbow is small | below r=8 default | **k90 = 200/576**; floors ~1.0, no sound elbow | ❌ Reversed |
| 4 | Loss-elbow agrees with k90 | agreement | **only within-representable**: A 4=4 ✓; B mirage (elbow 2) | ⚠️ Conditional |

**Row 3 reversed the prediction.** The controlled task proves the method: elbow = 4 = the true known rank, and the r=2→4 transition is a 3-order drop. But the *real* cross-layer delta `W_Q1 − W_Q0` is **~200-dimensional (≈35% of 576) — not low-rank by construction**. LoRA's max r=16 (rank ≤ 16) cannot represent it, so no floor exists and the elbow degenerates to a false 2.

**Why it matters:** a *fine-tuning* delta (what LoRA is for) is a small perturbation of one weight; `W_Q1 − W_Q0` is the difference between two separately-trained layers — structurally dissimilar, not a nudge. The measurement caught it before any real sweep could save the premise.

**The gatekeeper rule (the takeaway).** The spectrum gates the method: if effective `k90 ≤` sweep-range the loss-elbow finds it exactly; if `k90 ≫` range the delta isn't LoRA-adaptable — read the spectrum *first*, sweep *only when it fits*. The loss floor alone cannot certify an elbow (Experiment C's two panels make the difference unmistakable).

**Status:** Experiments A (elbow 4 = true rank), B (k90 = 200, not low-rank), C (two-panel elbow plot; both regimes) measured.

**The mechanism in three sentences**

The loss-vs-rank curve is really the delta's spectrum in disguise. The best rank-r approximation error of a delta equals the tail of its squared singular values from σ_{r+1} on: loss(r) = Σ_{i>r} σᵢ². So "sweeping r" is just walking along the spectrum's cumulative tail.

Controlled A has a spectral cliff: its delta is exactly rank 4, so σ₅…σ₅₇₆ = 0. The tail is 0 for any r ≥ 4, and huge for r < 4. Under r<4 you're boxed below the capacity that could touch the signal → floor pinned at 3e-3. The instant r=4 grants the capacity, the fit becomes exact → loss collapses to the optimizer floor (2.9e-6) and then goes flat (r=8,16 add nothing). The elbow is that cliff in representational capacity.

Real (B) has a long spectral tail, no cliff. Its delta is ~rank-200 (k90=200). For every r in {2…16}, r ≪ 200, so you're always on the smooth monotone tail — each extra rank nets a bit more captured variance (~1.2×), but none ever crosses the point where the tail hits zero. No capacity cliff exists in the sweep, so no elbow — just a slow monotone slide. The flagged r=2 is a local derivative wobble, not a phase change.

**The mechanism, precisely.**

- **Under-parameterized (r < true rank):** the model cannot represent the delta — the loss floor is a capacity bound (the best rank-r approximation), and no amount of optimization fixes it.
- **At/over-parameterized (r ≥ true rank):** exact fit is reachable — the floor becomes the optimizer's noise (~1e-6), not a capacity wall.
- **The elbow** is the transition between those two regimes: the point where the capacity you give exactly meets the rank the data demands. That's why elbow = k90 when the sweep contains it — it's the same number seen two ways (Unit 1's spectrum, and this unit's training curve). And it's why the spectrum gate exists: you can read that cliff before paying for any sweep.

---

## Unit 5 — Spectra at Scale: does low rank survive 135M → 1.7B? (scaffolded)

**Question.** Unit 1 measured the spectrum of one model. Does the rank structure *travel* as the model grows? Three SmolLM2 sizes (135M, 360M, 1.7B), the same five matrices, one SVD each, plotted against parameter count.

**Hypothesis.** If the effective rank tracks the model's *width* (d = 576 → 960 → 2048), k90/n should stay roughly flat across sizes — concentration is a property of the architecture, not the scale. If it tracks something else, the ratio moves and the "concentration is intrinsic" story needs rewriting.

**The measurement.** Load each model fresh, extract the five matrices, SVD each to the 90%-energy rank (`k90`) and top-1 energy share, then `del model` + `gc.collect()` before the next size. The download is the constraint: 360M (~720MB) and 1.7B (~3.4GB) will stream into the HF cache once; after that every run is local. CPU time is bounded by the 1.7B SVD (only the top of the spectrum is needed — `torch.linalg.svd(..., full_matrices=False)`).

**Findings:** _pending measurement._ To be filled after the notebooks run.
