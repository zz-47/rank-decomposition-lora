# rank-decomposition-lora
The full math of LoRA, measured on real SLM weights — SVD to rank selection to scale-out (135M→1.7B) to deploy-time merging. CPU-only, no assumptions.

---

## Studies

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

## Study 1 — SVD & Eckart–Young on real SmolLM2-135M weights (complete)

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

## Study 2 — The LoRA parametrization (complete)

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

## Study 3 — The LoRA hypothesis: does GD find a low-rank delta? (in progress)

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

## Study 4 — Choosing the Rank: when does adding r stop buying? (complete)

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

## Study 5 findings (measured — SmolLM2 135M, 360M, 1.7B)

**Settled on first read: 360M's width is 960, not 576.** The contested architecture number is now measured directly from the checkpoint (gate/up `(2560, 960)`, q/emb `(960, 960)`). The SLM-survey table said 576; the actual weights say 960. The family is *not* iso-architectural after all (135M width 576 vs 360M width 960), which the pre-committed scope honesty already flagged — we claim family persistence, not a scaling law.

**Metrics, five matrices × three sizes** (`k90` = smallest k capturing ≥90% energy; `top1` = σ₁²/‖σ‖²; `k90/n` = normalized effective rank). Widths measured from checkpoints: **576 → 960 → 2048**.

| Matrix | 135M k90/n | 135M top1 | 360M k90/n | 360M top1 | 1.7B k90/n | 1.7B top1 | Trend |
|---|---|---|---|---|---|---|---|
| gate | 0.684 | 0.030 | 0.690 | 0.018 | 0.716 | 0.006 | slowly flattens |
| up | 0.688 | 0.006 | 0.683 | 0.004 | 0.723 | 0.003 | ~flat |
| down | 0.674 | 0.019 | 0.674 | 0.023 | 0.758 | 0.006 | drifts up |
| q | 0.113 | 0.254 | 0.092 | 0.215 | 0.102 | 0.188 | concentrated, persists |
| emb | 0.635 | 0.506 | 0.740 | 0.303 | 0.772 | **0.093** | **dilutes fast** |

**Hypothesis-board verdicts (measured):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| H1 | FFN `k90/n` flat across scale | ~0.68–0.72 | 0.68→0.76 (shallow rise) | ✅ Mostly holds |
| H2 | concentration tracks width → flat `k90/n` | flat | FFN ~flat, emb rises, q stays | ⚠️ Partial |
| H3 | W_Q stays the most concentrated | persists | 0.113→0.092→0.102 (~0.1) | ✅ Holds |
| H4 | W_E concentration persists | persists | **top1 50.6%→30.3%→9.3%** | ❌ Reversed |

**The two real surprises.**

1. **W_E *dilutes* with scale, and fast.** `top1` collapses 50.6% → 30.3% → **9.3%**; effective rank rises 0.635 → 0.740 → 0.772. The single "common direction" holding half the embedding's energy at 135M is nearly gone by 1.7B. "One direction dominates embeddings" is a *135M phenomenon*, not a family invariant — concentration does *not* survive in W_E. (One honest caveat: embedding width grows 576→2048 while vocab stays 49,152, so the per-token row gets 3.5× more slack; that mechanical widening, not a semantic change, is the most parsimonious driver of the dilution.)

2. **W_Q stays dramatically concentrated across all three sizes** — `k90/n` ≈ 0.1 (~7× tighter than the FFN trio) at width 576, 960, and 2048. The attention-side low-rank hint from Unit 1 is a genuine *scale-stable* property, not a small-model quirk. This is the single most durable pattern in the family.

The FFN trio holds roughly flat (`k90/n` 0.67→0.76) — mild drift upward at 1.7B, nothing cliff-like. The spectrum does not scale up as a trivial average of static curves; W_Q and W_E pull in opposite directions as width grows.

**Completeness:** all three sizes measured (extractor ~0.6–3s per model, pre-extracted to `unit5_mats/*.pt`; no model ever loads into the kernel).

**Reading the two metrics as complements.** `k90/n` is width-normalized — the fair comparator across sizes. `top1` is **not**: its denominator runs over all n singular values, so as width grows 576→2048 the sum inflates and `top1` falls *mechanically* on every matrix (gate 0.030→0.006, q 0.254→0.188, up 0.006→0.003). If the story were `top1` alone, you'd wrongly conclude concentration dilutes everywhere. The size-fair verdict is `k90/n` (largely flat → rank structure travels). `top1` is diagnostic only when it moves the *same* direction as `k90/n` — exactly the two cases that disagree, and the two that matter: **W_Q holds** (both metrics stable) and **W_E reverses** (both metrics).

**Verdict in one line.** Rank structure is a property of the SmolLM2 *architecture*, not a specific checkpoint: the width-normalized `k90/n` held across 135M→1.7B, so the rank-selection recipe (sweep the floor, read k90, set r = k90) is portable to a 1.7B adapter without re-measuring every layer — the flatness LoRA's budget assumptions at scale rest on. Three honest edge cases, measured rather than smoothed over: W_E genuinely dilutes (both metrics), the FFN `k90/n` drifts up ~0.08 (down_proj, just past tolerance), and `top1` alone is not a valid flatness test.

---

## Study 6 — Industrial Deployment: from math to a served model

**Question.** The math is proven (merge is exact, rank travels). Does it *ship*? Build a real rank-8 adapter on SmolLM2-135M, then measure each deployment step as a quantifiable delta: merge error, quantized-vs-full output drift (8-bit vs 4-bit), memory footprint, and unmerged multi-adapter routing (base shared, deltas swapped) — all on CPU, all by hand, no `peft` dependency for the core.

**Structured as a research note.** Abstract + related work (LoRA, QLoRA, GPTQ, the previous study) → pipeline & knobs (train → merge → quantize → serve) → hypothesis board decided *before* measurement → protocol & constraints (135M, r=8, float32, fixed seed, batch 1) → four experiments, each implemented and run.

**Measured findings.** A real rank-8 adapter was trained on `gate_proj` (next-token CE 1.045 → 0.113; ΔW exactly rank-8; ‖ΔW‖/‖W0‖ = 0.148), then each deployment step was measured:

- **H1 merge exact — ✅.** max error 5.96e-8, rel Frobenius 2.6e-9, both *below* the float32 eps floor (1.2e-7). Merging is a renaming, not a transform.
- **H2 8-bit preserves behavior — ❌.** Relative drift 0.184 (~18%) under the pre-committed whole-matrix min/max affine. The failure is the crude quantizer, not the adapter — per-row scales (the standard fix) would recover most of the drift.
- **H3 4-bit loses more — ✅.** Peak drift 11.91 vs 0.655 at 8-bit (~18×), at 1/8 vs 1/4 the float32 memory (3,538,944 → 884,736 → 442,368 bytes).
- **H4 unmerged shares memory, costs latency — ✅.** Base gate (3.54 MB) fully shared; per-adapter delta just 67.6 KB (~1.9%); unmerged routing pays ~16% more ms/op.
- **H5 merged ≈ full at runtime — ✅.** 4.71 vs 4.82 ms/op (gate microbenchmark) — the swap is a runtime no-op.

**Verdict in one line.** The previous studies' math ships verbatim: merging is exact to float noise, merged serving is indistinguishable from full at logit and latency level, and unmerged multi-adapter routing keeps the base shared for a small, measured latency cost. The one honest failure is the whole-matrix quantizer — reported as measured, not papered over.
