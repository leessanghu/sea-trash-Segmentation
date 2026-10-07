# Coastal Marine Debris Segmentation — 2026 National Park Satellite Monitoring AI Challenge

A model that finds marine debris accumulated on coastlines in 0.3 m satellite imagery, 256×256 patches.
Submission for **Topic 4 (Coastal Debris Detection and Scale Estimation)** of the 2026 National Park
Satellite Monitoring AI Challenge.

| | |
|---|---|
| Final rank | **29th / 138 teams** |
| Public score | **0.7195** (1st place: 0.88) |
| Evaluation | Automated inference — teams submit model + inference code, organizers score on a private held-out set |
| Submission | 6-model ensemble (2 CNN + 4 ViT), TTA×8, fixed post-processing |

---

## 1. The task

**Input:** 0.3 m/px RGB satellite patch, 256×256. **Output:** binary debris mask (RLE-encoded).

**Metric**

```
final score = 0.5 × macro F1  +  0.5 × shape score
```

- **macro F1 (patch-level presence)** — a predicted mask with **≥50 positive pixels** counts as a positive
  prediction for that patch. F1 is computed separately for the positive and negative classes and averaged.
- **shape score (extent)** — for every patch that actually contains debris, precision/recall are computed
  pixel-wise with a **1-pixel tolerance** at boundaries, then combined via harmonic mean. A patch with no
  positive prediction scores 0.

This two-tier metric was the central design constraint: **whether debris is present** and **how well its
extent is drawn** are scored separately, and different models turned out to be good at each.

---

## 2. The core difficulty — training and evaluation data come from different platforms

The competition provided no training data. The reference dataset organizers pointed to, AIHub's
"National Park Change Detection" dataset, only had coastal-debris labels on **0.1 m drone imagery** —
and the organizers themselves warned that "the platform and resolution differ substantially; do not
treat this as resolution-matched training data."

| | Available for training | Evaluation target |
|---|---|---|
| Platform | Drone | Satellite |
| Resolution | 0.1 m | 0.3 m |
| Sharpness | Crisp | Blurred by sensor MTF |
| Location | Mostly one site (Taean) | Undisclosed (national-park coastline) |

Because of this gap, **local validation scores did not predict the leaderboard.** The most valuable
record in this repo isn't the model architecture — it's how that mismatch was handled (see §6).

---

## 3. Data

Drone-only labels weren't enough, so openly-licensed aerial/satellite imagery was collected and used to
build synthetic training data. The final training set totals **74,053 images**.

| Set | Count | Content |
|---|---|---|
| `real_pos_train` | 146 | AIHub drone positive patches, resampled to 0.3 m |
| `real_neg_train` | 1,855 | Negative patches, same source |
| `real_val` | 911 | Validation (Taean, site DR2001) |
| `synth_train` | 30,000 | Debris fragments composited onto backgrounds (copy-paste) |
| `cp_typed_train` a/b/c | 22,540 | **Type-specific generation** — waterline bands, wrack lines, sand piles, rock piles, scatter, structure bands, styrofoam piles, quay/pier debris, etc. |
| `bg_naip_train` | 3,855 | US NAIP aerial, 0.3 m coastal backgrounds (Public Domain) |
| `bg_ports_train` | 8,452 | Port/harbor backgrounds |
| `bg_coast2_train` | 1,357 | NZ LINZ, HK LandsD coastlines |
| `bg_hawaii_train` | 1,075 | Hawaii coastline |
| `real_landneg_train` | 1,631 | Inland backgrounds that trigger false positives |
| `mudflat_nl` | 380 | Dutch tidal flats (false-positive source) |

**Rules followed when generating synthetic data**

- Positives were only pasted onto **tiles that actually contain water**. NAIP has a near-infrared band,
  so its water mask was reliable.
- Port tiles were only used when water coverage was ≥15% — otherwise concrete and tidal channels got
  misread as water.
- Nothing was pasted onto foam, decks, white boats, or pontoons.
- Pasted objects were randomized in angle/scale/brightness, and degradation mimicking a 0.3 m satellite
  sensor (chroma downsampling, resampling) plus random histogram matching were applied.

**External-data licensing** — only sources that allow commercial use and model transfer were used.
Map-service tiles (Google/Naver/Kakao/VWorld) were excluded because their terms forbid use in AI training.

---

## 4. Model

Six differently-structured models, trained on the same data, with **their predicted probabilities
averaged**. Every member gets 8-way D4-symmetric test-time augmentation.

| # | Architecture | Input handling | Note |
|---|---|---|---|
| 0 | UNet + ResNet18 | 256 → 512 (2× upscale) | Hard-negative fine-tuned |
| 1 | UNet + ConvNeXt-T | 256 → 512 | |
| 2 | DINOv2 ViT-S/14 + multi-layer decoder | 448 | Hard-negative fine-tuned |
| 3 | DINOv3 ViT-S/16 + multi-layer decoder | 448 | |
| 4 | DINOv3 ViT-S/16 + **LoRA** (r=16) | 448 | Backbone frozen, only low-rank adapters trained |
| 5 | DINOv2 ViT-S/14 + **DoRA** (r=16) | 448 | same idea |

**Design rationale**

- **2× upscaled input for the CNNs** — debris clumps are only a few pixels wide at 0.3 m. Upscaling the
  input raises effective resolution.
- **ViTs fuse 4 intermediate layers** (2, 5, 8, 11) instead of just the last one, which loses spatial detail.
- **LoRA/DoRA members** — keep pretrained features mostly intact so they're less rattled by the
  drone→satellite domain gap, and make different mistakes than the fully fine-tuned members (ensemble
  diversity). Adapters are merged into the base weights at save time, so inference code treats them like
  any other model.
- **Why mix CNN and ViT** — leaderboard evidence showed **CNN share pushed up F1 (presence), while ViTs
  pushed up shape (extent)**. Leaning too hard either way cost points.

### Hard Negative Mining (HNM)

Failure analysis showed the single biggest source of loss was **confidently-wrong false positives**
(77 out of 795 negative patches).

1. Run the current 6-member ensemble over 1,855 negative training patches.
2. Pick the **400** highest-confidence false positives into a dedicated hard-negative set.
3. Mix it in at 15% and fine-tune briefly — **2 epochs × 200 steps, 0.3× learning rate**.
4. Keep only the members that actually improved.

Swapping in the fine-tuned ResNet18 and DINOv2 members moved the score **0.7079 → 0.7195 (+0.0116)**.
That's the final checkpoint set in this repo.

---

## 5. Inference and post-processing

```python
# 1) Average 6 members × TTA8 → probability map prob(256×256), classifier-head probability cls
# 2) Patch-level decision
decision_score = 0.25 * cls + 0.75 * (mean of top-50 probability pixels)
if decision_score < 0.40:  → empty prediction
# 3) Mask
mask = prob > 0.5
mask = remove connected components smaller than 10 px
if mask.sum() < 50:  → fill with the top-80 highest-probability pixels   # a positive decision must leave ≥50px
mask = dilate by 1px (3×3)
```

Fixed config: `t_pix 0.5 / t_img 0.4 / alpha 0.25 / min_cc 10 / rescue_px 80 / dilate 1`

**Why each post-processing step exists**

- **Decoupled thresholds for decision vs. mask** — missing a positive patch scores 0 on *both* F1 and
  shape, 3–5× costlier than a false positive. So the decision threshold is kept loose (0.40) while the
  mask threshold stays conservative (0.50).
- **`rescue_px 80`** — if the decision says "positive" but the mask has fewer than 50 pixels, the patch is
  rules-wise an empty prediction and loses points on both metrics. Filling it with the top-80 probability
  pixels prevents that.
- **`dilate 1`** — the metric forgives 1-pixel boundary error, so a 1-pixel dilation **raises recall at
  essentially no precision cost**. Measured leaderboard gain: +0.002.

---

## 6. Validation strategy and what went wrong

This is where most of the competition's time went. **Local validation (drone, a single site in Taean)
repeatedly disagreed with the leaderboard.**

### Full leaderboard history

| Submission | Change | Public | Local Δ | Outcome |
|---|---|---|---|---|
| v4 post-processing | | 0.6596 | | |
| v5 + val-tuned post-processing | `tune.py`-optimized locally | **0.6418** | best locally (0.8223) | ❌ local-best was LB-worst |
| v5 + fixed post-processing, dilate 0 | | 0.7059 | | |
| v5 + fixed post-processing, dilate 1 | | 0.7079 | | ✅ baseline |
| hnm-a | swapped 3 members incl. ConvNeXt | 0.7073 | +0.0094 | ❌ |
| **hnm-b** | swapped ResNet18 + DINOv2 | **0.7195** | +0.0071 | ✅ **final** |
| cnxt removed | dropped the locally-worst single member (5 members) | 0.6978 | +0.0064 | ❌ **−0.022** |
| 12-member expansion | added 6 newly-trained, mostly-ViT members | 0.7077 | +0.0118 | ❌ |

### What this taught

1. **Post-processing mattered more than the model.** With identical weights, changing only
   post-processing moved the score 0.6418 ↔ 0.7079 — a **0.066** swing, more than 5× the largest gain
   from a model change (+0.0116).

2. **Never tune post-processing on an out-of-distribution validation set.** The locally-optimal setting
   was the worst on the leaderboard. Post-processing was fixed after this and never touched again. This
   matches Ovadia et al. (NeurIPS 2019): calibrating on an i.i.d. validation set can make calibration
   *worse* under distribution shift.

3. **Don't drop ensemble members based on local scores.** Removing ConvNeXt — the weakest single member
   locally — improved local score by +0.006 but cost **−0.022** on the leaderboard. That member wasn't
   weak locally; it was weak specifically at *shape*, while it was carrying the *decision* signal on real
   satellite data. CNNs and ViTs with different pretraining sit on different "local-score ↔ test-score"
   lines, so comparing the two families by local score alone doesn't even make sense in principle
   (Miller et al., ICML 2021; Andreassen et al., TMLR 2022).

4. **Decision and shape are won by different models.** Raising CNN share in the ensemble pushed F1 from
   0.924 to 0.949 but dropped shape from 0.640 to 0.610 — the two effects canceled in the combined score,
   which hid the pattern. Across 5 leaderboard submissions, the score tracked **CNN share in the decision
   ensemble**, not the overall model mix.

5. **So the final architecture averages probabilities twice from a single forward pass** — a
   **CNN-weighted ensemble decides presence**, a **shape-validated ensemble draws the mask**
   (`predict_probs_dual`, `postprocess_dual`). Keeping the decision weights unchanged means patch-level
   decisions stay 100% identical to a known-good submission, so F1 is fixed and only shape can move. The
   same idea — a high threshold for decision, a lower one for the mask — won the SIIM-ACR Pneumothorax
   Kaggle competition.

---

## 7. Repository layout

```
.
├── predict.ipynb          Submission inference notebook (writes prediction.csv per spec)
├── requirements.txt
├── knps/                  Inference modules
│   ├── model.py           Model definitions, checkpoint loader, TTA ensemble inference
│   ├── postprocess.py     Decision / mask post-processing
│   ├── rle.py             RLE encoding
│   └── config.example.json  Final submission config (weight paths, post-processing values)
└── training/              Training code
    ├── train.py           Single-member training (LoRA/DoRA/SoMA, augmentation, EMA)
    ├── run_retrain.py     Full ensemble retraining pipeline
    ├── hnm_pipeline.py    Hard negative mining
    ├── make_typed_aug.py  Type-specific synthetic data generation
    ├── exp_eval.py        Evaluation under fixed post-processing
    ├── soup.py            Weight averaging (model soups)
    └── colab/build_kaggle_notebook.py   Generates the Kaggle T4×2 training notebook
```

## 8. Running it

This repo contains **code only** — the trained weights (6 files, 541 MB total) are not included.
The architecture lives in `knps/model.py` and the exact config in `knps/config.example.json`, so the
ensemble can be reproduced with the scripts in `training/`.

```bash
pip install -r requirements.txt

# if you have weights, place them as assets/model/model0.pt ... model5.pt
cp knps/config.example.json assets/model/config.json

# inference (input patches expected at ./input/images/*.png)
jupyter nbconvert --to notebook --execute predict.ipynb
```

Training targeted Kaggle T4×2. `training/colab/build_kaggle_notebook.py` generates a notebook that
splits members across both GPUs.

---

## 9. License and attribution

- **Training/inference code** (`training/`, `knps/`) — by this repo's author.
- **`predict.ipynb` and the baseline weight lineage** — derived from the baseline package distributed by
  AIFactory (competition organizer). Original license terms are kept verbatim in
  [`LICENSE-baseline.txt`](LICENSE-baseline.txt); component provenance is in [`NOTICE`](NOTICE).
- **Built with DINOv3.** Members 3 and 4 are initialized from Meta's DINOv3 pretrained weights
  (`timm/vit_small_patch16_dinov3.lvd1689m`) and follow the
  [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
- **DINOv2** (members 2, 5) — Apache 2.0. **segmentation_models_pytorch** — MIT.
  **ResNet18 / ConvNeXt ImageNet weights** — BSD-3-Clause / Apache 2.0.
- External imagery used for training was limited to sources that allow commercial use and model transfer
  (US NAIP — Public Domain, NZ LINZ — CC BY, HK LandsD, etc.). See `NOTICE` for full provenance.
