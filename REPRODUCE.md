# Reproducing ProtoMask-Set

This artifact reproduces the paper at three tiers of cost. Everything is read-only on frozen
public models — **nothing is fine-tuned** except the small audited decoder head (a deliberate
negative result). Pick the tier that matches what you have.

## Seen/novel splits (shipped)

The seen/novel splits are deterministic functions of an integer seed and are **shipped under
[`splits/`](splits/)** (`ade20k_seed{0..4}.json`, `voc_seed{0..4}.json`, with `SHA256SUMS`).
They are byte-identical to the RNG the experiment scripts use; regenerate and checksum them with:

```bash
python scripts/dump_splits.py            # numpy only; rewrites splits/ + SHA256SUMS
(cd splits && sha256sum -c SHA256SUMS)   # verify; the manifest uses bare names, so run it from splits/
```

`make_split()` in `scripts/run_calib_baselines.py` reads these files when present and falls back
to the identical RNG otherwise, so the released files are authoritative.

## Tier A — regenerate the LaTeX tables from shipped result JSONs (numpy only)

No datasets, models, or GPU. Reads the per-run JSONs under [`results/`](results/) and rewrites the
`tables/*.tex` they back (see [`results/manifest.json`](results/manifest.json) for the
table → JSON → script map):

```bash
python scripts/finalize_tables.py    # tab_real_main, tab_real_diffuse, tab_real_decoder
```

The calibration / edge / regime / VOC / baseline tables are transcribed from the `summary` blocks
of the corresponding shipped JSONs (`metrics_calib`, `metrics_regime`, `metrics_voc`,
`metrics_stage3`, `metrics_stage4`); `manifest.json` records which JSON backs each.

## Tier B — re-run the cheap CPU/numpy calibration study from a cached feature set

Needs the one-time feature cache (`cache/<split>/*.npz` + `text_protos.npy`/`qfeat.npy`) produced
in Tier C. No GPU. From the cache it recomputes every calibration/collapse number:

```bash
PYTHONPATH=src PMX_CACHE=cache python scripts/run_calib_baselines.py --cache cache --out-dir results
PMX_CACHE=cache python scripts/pm_stage2.py   # dense-CLIP regime (tab_real_regime)
PMX_CACHE=cache python scripts/pm_stage3.py   # graph-Laplacian / trust-region / AURC (tab_real_baselines)
PMX_CACHE=cache python scripts/pm_stage4.py   # OV-aware text-side calibrator
PMX_CACHE=cache python scripts/pm_adaece.py   # adaptive-bin / class-balanced ECE
```

## Tier C — rebuild everything from raw data (GPU)

Recompute the feature caches and dense-CLIP/VOC runs from the raw datasets and frozen models. See
[`MODELS.md`](MODELS.md) for exact model identifiers and the external SCLIP checkout. All roots are
environment variables with portable repo-relative defaults, so a missing override fails locally rather
than against an author-specific path:

| Variable | Meaning | Default |
|---|---|---|
| `PMX_CACHE` | feature/output cache dir | `cache` |
| `PMX_SCLIP` | external SCLIP checkout (dense-CLIP CSA) | `SCLIP` |
| `PMX_ADE_ROOT` | `ADEChallengeData2016` root | `data/ADEChallengeData2016` |
| `PMX_VOC_ROOT` | `VOCdevkit/VOC2012` root | `data/VOCdevkit/VOC2012` |
| `PMX_ADE_NAMES` / `PMX_VOC_NAMES` | class-name lists | `$PMX_SCLIP/configs/cls_{ade20k,voc20}.txt` |

The headline accuracy tables are built by passing the **full** ADE20K-150 validation split
(`--n-images 2000`); the *evaluated* set is the SAM-cacheable subset of it (see the honesty note
below), not 2000 images. They also use a
**600**-image train cache for the calibrators, both built with the weak masked-bbox region encoder
(`--encoder bbox`); the script defaults (`--n-images 400`, `--encoder context`) are *not* the
headline configuration, so pass the flags below explicitly.

```bash
# masked-bbox SAM+CLIP feature cache for the headline tables (one-time GPU bottleneck)
PYTHONPATH=src PMX_CACHE=cache python scripts/cache_features.py --split validation --n-images 2000 --encoder bbox --ade-root $PMX_ADE_ROOT
PYTHONPATH=src PMX_CACHE=cache python scripts/cache_features.py --split train      --n-images 600  --encoder bbox --ade-root $PMX_ADE_ROOT
# stronger context-padded encoder for the robustness probe (tab_real_encoder); build into a separate cache
PYTHONPATH=src PMX_CACHE=cache_ctx python scripts/cache_features.py --split validation --n-images 2000 --encoder context --pad 0.25 --ade-root $PMX_ADE_ROOT
PYTHONPATH=src PMX_CACHE=cache_ctx python scripts/cache_features.py --split train      --n-images 600  --encoder context --pad 0.25 --ade-root $PMX_ADE_ROOT
# competitive training-free dense-CLIP (needs SCLIP)
PMX_SCLIP=/path/to/SCLIP PMX_ADE_ROOT=/path/to/ADE python scripts/pm_stage1.py
PMX_SCLIP=/path/to/SCLIP PMX_VOC_ROOT=/path/to/VOC python scripts/voc_collapse.py
```

## Environment

See [`requirements.txt`](requirements.txt). Tiers A–B need only `numpy` (+ `matplotlib` for figures);
Tier C additionally needs `torch`, `transformers` (frozen SAM/CLIP), and, for the dense-CLIP regime,
the OpenAI `clip` package and the external SCLIP repo. `requirements.txt` lists floors that are known
to work; the reported GPU runs used the recorded environment below. Tiers A–B are insensitive to
these versions, and the qualitative Tier C findings hold across nearby releases.

| Package | Reported runs |
|---|---|
| `torch` | 2.4.0 (NVIDIA NGC 24.07 build) |
| `torchvision` | 0.19.0 |
| `transformers` | 4.46.3 |
| `numpy` | 1.24.4 |

## Honesty notes

* Tiers A–B are fully portable and dataset-free / cache-only. Tier C requires the public datasets,
  the frozen models in `MODELS.md`, an external SCLIP checkout, and a GPU.
* The dense-CLIP/VOC numbers are reported at **patch resolution** in the tables and at **full
  resolution** in the text (e.g. ADE 14.5 vs 16.5 mIoU, VOC 75.3 vs 81.8); both are stated in the paper.
* `smoke_test.py` exercises the model/metric/table-writer paths end-to-end on synthetic data with no
  external dependencies.

## Independent reproduction (H100)

We re-ran the masked-crop pipeline end-to-end on an H100 (rebuilt the SAM+CLIP cache, then re-ran the
diffusion sweep, the trained decoder, and the calibration comparison over the five seen/novel splits).
Every qualitative finding reproduced: a global temperature is the best overall calibrator, parameter-free
diffusion is best on novel-class ECE, per-class temperature and vector scaling both collapse novel-class
IoU (to about 2) while inflating novel-class ECE (to about 28 and 32), and class-symmetric diffusion
preserves novel-class IoU (about 9.5) and ties the baseline on mIoU. The only quantitative difference is
the absolute masked-crop mIoU, which reproduces near 10.2 versus the 8.9 of the committed run because SAM
automatic mask generation is stochastic across cache builds. We therefore keep the committed numbers and
report this re-run as an independent confirmation rather than overwriting them.

## Extended experiments

Six further stages, all driven from the same dense-CLIP cache (`pm_stage1.py`) and the
shipped seen/novel splits. Set `OMP_NUM_THREADS=16` on many-core machines: torch's default thread
count (one per core) makes these scripts up to 5x slower with bit-identical output.

```bash
PMX_CACHE=cache python scripts/pm_stage5_align.py     # alignment-preserving heads (B1-B6) + frontier -> metrics_align.json
PMX_CACHE=cache python scripts/pm_stage7_diag.py      # drift diagnostics + confound ablation            -> metrics_diag.json
# LPOSS (authors' code, verbatim) under our protocol + stacked-head audit
python scripts/lposs_extract.py --lposs /path/to/LPOSS --out scripts/lposs_core_auto.py
PMX_CACHE=cache PMX_ADE_ROOT=... python scripts/pm_stage8_lposs.py   # needs faiss-cpu + cupy-cuda12x + torch.hub DINO
# COCO-Stuff -> ADE20K transfer (fit on COCO val2017 patches, evaluate on the ADE cache)
PMX_CACHE=coco_cache PMX_ADE_NAMES=$PMX_SCLIP/configs/cls_coco_stuff.txt \
  PMX_COCO_IMG=data/coco/val2017 PMX_COCO_ANN=data/coco/stuffthingmaps/val2017 \
  python scripts/pm_stage9a_cache_coco.py
PMX_CACHE=cache PMX_COCO_CACHE=coco_cache python scripts/pm_stage9_transfer.py  # -> metrics_transfer.json
PMX_CACHE=cache python scripts/pm_stage10_cost.py     # runtime/memory table                        -> metrics_cost.json
python scripts/finalize_ext_tables.py --in-dir results/extended --out-dir tables
python scripts/make_ext_figs.py                        # fig_frontier.pdf, fig_drift.pdf
```

Notes on the reruns:

* `pm_stage8_lposs.py` extracts the LPOSS propagation functions byte-for-byte from the authors'
  repository (SHA-256 per function printed by `lposs_extract.py`) and shims only `faiss.knn_gpu`
  with an exact torch top-k, because the prebuilt faiss GPU wheels ship no sm_90 (H100) kernels.
* The evaluated masked-crop set is the number of CACHED images, not `--n-images`:
  `cache_features.py` skips images where SAM returns fewer than two masks (1623 of 2000 val images
  in the rebuild). `finalize_tables.py` now reads the true count from `metrics_calib.json`.
* A second independent rebuild reproduced the dense-CLIP tables as follows, derived
  by `python3 scripts/repro_deviation.py` into `results/extended/repro_deviation.json`: every
  parameter-free row (uncalibrated / global T / per-class T / diffusion / Laplacian / LPOSS) agrees
  to within 0.05, and the stochastically trained decoder rows to within 0.85 (largest single
  disagreement: PASCAL VOC novel-class ECE 16.69 vs 17.54; ADE20K maximum 0.40). The rebuild records
  AURC as NaN for the two trained-decoder rows of `tab_real_baselines`, so those two values are not
  compared. The masked-crop findings reproduced with baseline mIoU 10.2, matching the pre-registered
  rebuild value above.
* LPOSS provenance is persisted, not just printed: `python3 scripts/lposs_provenance.py --lposs
  /path/to/LPOSS` writes/verifies `results/extended/lposs_provenance.json`, which holds the per-function
  SHA-256 of the four extracted propagation functions, the upstream commit they match, and the one
  documented deviation (faiss.knn_gpu -> exact torch top-k).
