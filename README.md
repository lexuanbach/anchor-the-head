# Anchor the Head: Alignment Drift in Open-Vocabulary Segmentation

Code, seen/novel splits and result records for the WACV 2027 paper
*Anchor the Head: Alignment Drift in Open-Vocabulary Segmentation* (Xuan-Bach Le).

The paper audits small correction heads trained on the seen classes over frozen open-vocabulary
segmentation stacks (SAM+CLIP masked crops and training-free dense CLIP). Seen-only, class-specific,
unanchored corrections sharply reduce novel-class IoU. Anchoring the same head to the frozen logits,
or making the correction class-symmetric, preserves it. No foundation model is fine-tuned.

## Main results

- On the dense-CLIP stack (14.5 mIoU on ADE20K), a seen-only residual head takes novel-class IoU from
  15.1 to 0.1, and per-class calibrators behave the same way.
- Frozen-logit distillation at lambda=30 keeps novel-class IoU at 15.3 (collapse index 1.01) while
  raising mIoU from 14.5 to 15.8. A trust region at tau=0.5 recovers 14.6. Recovery is monotone in
  the anchor strength. Anchored heads still predict fewer novel pixels than the frozen base
  (7-11% vs 15.6%).
- Alignment drift tracks novel-class retention across 38 class-specific settings in 13 families
  (pointwise r=-0.90, cluster-bootstrap CI [-0.96,-0.74], family-level r=-0.65). Class-symmetric
  corrections retain novel classes even at large drift.
- A global temperature is the best calibrator (ECE 27.2 to 8.7 on dense CLIP) and leaves novel-class
  IoU unchanged.
- The pattern replicates on five bases: ADE20K masked-crop and dense-CLIP, LPOSS- and
  DINO-proxy-corrected dense CLIP, and PASCAL VOC.

Every reported number comes from a per-run JSON under `results/`. `results/manifest.json` maps each
table to its source JSON and script.

## Contents

- `src/protomask/`: frozen SAM+CLIP core (mask proposals, region and text embeddings, proposal graph,
  painting, metrics) and the trained graph decoder.
- `scripts/`: feature caching, calibration baselines, diffusion, decoder, dense-CLIP stages, VOC run,
  alignment-preserving heads, drift diagnostics, LPOSS comparison, COCO-Stuff transfer, cost, and the
  table and figure generators. `scripts/EXTENDED.md` maps the later stages to their outputs.
- `splits/`: seen/novel split files for ADE20K and VOC (seeds 0-4) with `SHA256SUMS`.
- `results/`: metrics JSONs for the masked-crop and dense-CLIP runs. `results/extended/` holds the
  alignment, drift, LPOSS, transfer and cost runs. `results/rebuild/` and `results/rebuild_mc/` hold
  the independent rebuild used for the reproduction bound.
- `REPRODUCE.md`: staged reproduction, environment variables and the rebuild log.
- `MODELS.md`: exact frozen model identifiers and revisions.
- `data/README.md`: dataset layout.

Generated tables (`tables/`) and figures (`figures/`) are written locally by the scripts and are not
tracked.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3 scripts/dump_splits.py                  # materialise seen/novel splits + SHA256SUMS (numpy only)
python3 scripts/finalize_tables.py              # regenerate the masked-crop tables from shipped JSONs
python3 scripts/finalize_ext_tables.py --in-dir results/extended --out-dir tables
python3 scripts/repro_deviation.py              # reproduction bound from the shipped rebuild records
# full reproduction from ADE20K / VOC on a GPU: see REPRODUCE.md
```

## Citation

```bibtex
@inproceedings{le2027anchor,
  title     = {Anchor the Head: Alignment Drift in Open-Vocabulary Segmentation},
  author    = {Le, Xuan-Bach},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## License

MIT, see `LICENSE`.
