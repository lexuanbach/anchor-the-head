# Reproducing Anchor the Head

The backbone models remain frozen; only the correction heads and calibrators are fitted.
`ProtoMask-Set`, the Python package `protomask`, and the `PMX_*` environment variables are
implementation names for the masked-crop and dense-stack experiments in this paper.

## Seen/novel splits

The artifact includes three distinct sets, each with five files and checksums:

- `splits/ade20k_seed{0..4}.json`: masked crops, `default_rng(seed).choice(150,30,replace=False)`.
- `splits/ade20k_dense_seed{0..4}.json`: dense ADE20K, five sequential `permutation(150)` draws
  from one `RandomState(0)` stream. The file suffix is the draw index, not an independent RNG seed.
- `splits/voc_seed{0..4}.json`: VOC, five sequential `permutation(20)` draws from `RandomState(0)`.

The first 30 ADE or 6 VOC permutation entries are novel. Masked-crop calibration reads the
masked-crop files; dense scripts reproduce the dense stream directly.

```bash
python scripts/dump_splits.py
(cd splits && shasum -a 256 -c SHA256SUMS)
```

## Tier A: regenerate from result records (no dataset or GPU)

```bash
python scripts/finalize_tables.py       # results/tables/: Tables 2, S2, S4
python scripts/finalize_ext_tables.py   # tables/: Tables 3, 5, 6, S6-S10
python scripts/make_calib_fig.py
python scripts/make_ext_figs.py
python scripts/drift_grouped_stats.py
python scripts/repro_deviation.py
```

These commands need numpy and, for plots, matplotlib. The manuscript has shorter captions and
layout adjustments, so numerical regeneration does not imply byte-identical LaTeX. S1 and S3
are written by the cache-based calibration script. Main Table 1 summarizes evaluation protocols and is maintained in the manuscript. Table 4 (VOC) and S5 (encoder comparison)
are transcribed from their JSONs. `results/manifest.json` maps every result table and figure.
The protocol and claims tables summarize multiple records rather than individual runs.

## Tier B: evaluate existing caches

Masked-crop calibration requires numpy and a masked-crop cache, with `text_protos.npy` and
`{train,val}/*.npz` containing region embeddings, labels and masks:

```bash
PYTHONPATH=src python scripts/run_calib_baselines.py --cache cache_calib --out-dir results_calib
```

Dense evaluation requires **CUDA**, torch and a different cache schema (`qfeat.npy`, patch
features and class logits). These scripts are not CPU-only:

```bash
PMX_CACHE=cache_dense python scripts/pm_stage2.py
PMX_CACHE=cache_dense python scripts/pm_stage3.py
PMX_CACHE=cache_dense python scripts/pm_stage4.py
PMX_CACHE=cache_dense python scripts/pm_adaece.py
```

Caches and dataset images are not distributed. Masked-crop and dense caches must use separate
directories; their records are incompatible. Defaults are `cache_mc` for the masked-crop cache
builder, `cache_dense` for dense ADE stages and `cache_voc` for VOC. Pass paths explicitly to
other masked-crop runners.

## Tier C: build caches from datasets (GPU)

See `data/README.md` for dataset layouts and `MODELS.md` for model identifiers and the pinned
SCLIP checkout. The dense scripts must import SCLIP's modified `clip` module.

```bash
export PMX_ADE_ROOT=data/ADEChallengeData2016
export PMX_VOC_ROOT=data/VOCdevkit/VOC2012
export PMX_SCLIP=SCLIP

# Original masked-crop accuracy protocol: attempt all validation images.
PYTHONPATH=src python scripts/cache_features.py --cache cache_mc --split validation --n-images 2000 --encoder bbox
PYTHONPATH=src python scripts/cache_features.py --cache cache_mc --split train --n-images 600 --encoder bbox

# Smaller calibration protocol: attempt 400 validation / 300 training images.
PYTHONPATH=src python scripts/cache_features.py --cache cache_calib --split validation --n-images 400 --encoder bbox
PYTHONPATH=src python scripts/cache_features.py --cache cache_calib --split train --n-images 300 --encoder bbox
PYTHONPATH=src python scripts/run_calib_baselines.py --cache cache_calib --out-dir results_calib

# Context-padded encoder comparison uses a separate small cache.
PYTHONPATH=src python scripts/cache_features.py --cache cache_ctx --split validation --n-images 400 --encoder context --pad 0.25
PYTHONPATH=src python scripts/cache_features.py --cache cache_ctx --split train --n-images 300 --encoder context --pad 0.25
PYTHONPATH=src python scripts/run_calib_baselines.py --cache cache_ctx --out-dir results_ctx

# Dense ADE20K and VOC caches.
PMX_CACHE=cache_dense python scripts/pm_stage1.py
PMX_CACHE=cache_voc python scripts/voc_collapse.py
```

`cache_features.py` supports both `--cache` and `PMX_CACHE` (the explicit option wins), and
both `--ade-root` and `PMX_ADE_ROOT`. It skips images with fewer than two SAM proposals.
The original calibration record has 347 cached validation images and 300 training images.
The commands specify attempted images, not a promise of exactly 347 cached images: the original
per-image cache manifest is unavailable and SAM cache builds can differ. The original accuracy
cache's image count was not recorded; the independent full rebuild cached 1623 of 2000 images.

## Extended dense experiments

Run the cached stages after Tier C; they require CUDA. `PMX_OUT` is supported where shown;
stages 2, 3, 4, 5 and 7 write their metrics into `PMX_CACHE`. Preserve the shipped records when
comparing reruns.

```bash
PMX_CACHE=cache_dense python scripts/pm_stage5_align.py
PMX_CACHE=cache_dense python scripts/pm_stage7_diag.py

# LPOSS: see requirements-lposs.txt and the upstream license in licenses/.
python scripts/lposs_extract.py --lposs /path/to/LPOSS --out scripts/lposs_core_auto.py
python scripts/lposs_provenance.py --lposs /path/to/LPOSS
PMX_CACHE=cache_dense python scripts/pm_stage8_lposs.py

# Transfer: fit on 600 COCO val2017 images, evaluate on ADE without fitting ADE labels.
PMX_CACHE=cache_coco PMX_ADE_NAMES="$PMX_SCLIP/configs/cls_coco_stuff.txt" \
  PMX_COCO_IMG=data/coco/val2017 PMX_COCO_ANN=data/coco/stuffthingmaps/val2017 \
  python scripts/pm_stage9a_cache_coco.py
PMX_CACHE=cache_dense PMX_COCO_CACHE=cache_coco python scripts/pm_stage9_transfer.py
PMX_CACHE=cache_dense python scripts/pm_stage10_cost.py

# DINO-proxy base and the corresponding audit.
PMX_CACHE=cache_dense PMX_PROXY_CACHE=cache_proxy python scripts/pm_stage11_proxy.py
PMX_CACHE=cache_proxy python scripts/pm_stage2.py
PMX_CACHE=cache_proxy python scripts/pm_stage5_align.py
# The latter two write metrics_regime.json and metrics_align.json in cache_proxy.
# Their released equivalents are results/extended/metrics_regime_proxy.json and metrics_align_proxy.json.
```

`lposs_provenance.py` requires an upstream checkout or source file, so invoking it without
arguments cannot overwrite the verified record. LPOSS uses DINO affinities, while the diffusion
and generic Laplacian controls use CLIP features. The GPU nearest-neighbor call is replaced by
exact torch top-k on these graphs. DINO's historical torch.hub source revision was not recorded;
this remains a limitation of bit-exact reproduction.

## Reproduction scope

The independent masked-crop rebuild confirms the qualitative pattern at baseline mIoU 10.2
versus 8.9 on the original cache. Per-class temperature and vector scaling reduce novel IoU to
about 1 (1.19 and 0.86), while diffusion retains about 9.5. These caches are not a matched
quantitative comparison.

`results/extended/repro_deviation.json` compares the available dense/VOC calibration and
smoothing records. Uncalibrated, global/per-class temperature, diffusion and Laplacian rows agree
within 0.05; trained-decoder metrics differ by at most 0.85 (VOC novel ECE; ADE maximum 0.40).
Temperatures are fitted calibrators. Trained-head AURCs are NaN in the rebuild and excluded.
There is no independent LPOSS, alignment/frontier, confound, or transfer rebuild record.
The ADE full-resolution 16.5 claim has no released record and is omitted from the revised paper;
VOC full-resolution 81.8 is recorded in `metrics_voc.json`.

`smoke_test.py` exercises model, metrics and table writing on synthetic inputs; it does not
validate benchmark performance. The released results were produced with torch 2.4.0,
torchvision 0.19.0, transformers 4.46.3 and numpy 1.24.4 (see `requirements-lock.txt`).
