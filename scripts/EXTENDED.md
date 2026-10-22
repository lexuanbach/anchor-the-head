# Extended experiments

The scripts behind the alignment-preservation, drift, LPOSS, transfer and cost results. All stages
read the dense-CLIP cache built by `scripts/pm_stage1.py` (see the top-level REPRODUCE.md) and the
shipped seen/novel splits. Outputs are in `results/extended/`.
Set OMP_NUM_THREADS=16 on many-core machines (5x speedup, bit-identical output).

| Script | Output JSON | Backs |
|---|---|---|
| pm_stage5_align.py | metrics_align.json | factorial and frontier tables, frontier figure |
| pm_stage7_diag.py | metrics_diag.json | confound-ablation table, drift figure |
| lposs_extract.py -> lposs_core_auto.py + pm_stage8_lposs.py | metrics_lposs.json | LPOSS table |
| pm_stage9a_cache_coco.py + pm_stage9_transfer.py | metrics_transfer.json | COCO-Stuff -> ADE20K transfer table |
| pm_stage10_cost.py | metrics_cost.json | cost table |
| pm_stage11_proxy.py | proxy_base.json | DINO-proxy base |
| drift_grouped_stats.py | drift_grouped.json | grouped and bootstrap drift statistics |
| finalize_ext_tables.py / make_ext_figs.py | -- | regenerate the tables and figures from the JSONs |
| lposs_provenance.py | lposs_provenance.json | SHA-256 digests behind the "verbatim" LPOSS claim |
| repro_deviation.py | repro_deviation.json | the reported-vs-rebuild reproduction bound |

`lposs_core_auto.py` is the verbatim AST extraction of the LPOSS authors' propagation.
`lposs_extract.py` only *prints* the per-function SHA-256, so `lposs_provenance.py` **persists** them
to `results/extended/lposs_provenance.json` together with the upstream repo, the commit whose
`segmentation/evaluation/lposs_eval.py` they match, and the single documented deviation. Re-verify
against an upstream checkout with `python3 scripts/lposs_provenance.py --lposs /path/to/LPOSS`.
`pm_stage8_lposs.py` shims only faiss.knn_gpu (exact torch top-k, since faiss wheels lack sm_90
kernels) and never edits the extracted code.

`repro_deviation.py` derives the reproduction bound from the shipped records: the compared uncalibrated, fitted-temperature and smoothing
dense-regime rows agrees to within 0.05, and the stochastically trained decoder rows to within 0.85
(largest single value: VOC novel-class ECE 16.69 vs 17.54, ADE20K maximum 0.40). AURC is NaN in the
rebuild for the two trained-decoder rows and is excluded.

The masked-crop regime rebuild JSONs are prefixed `mc_`.

This comparison does not include an independent LPOSS or alignment/frontier/diagnostic/transfer rebuild.
See REPRODUCE.md for the DINO-proxy audit commands and output locations.
