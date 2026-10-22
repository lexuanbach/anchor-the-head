"""Compute the ACTUAL maximum deviation between the published result records and the
independent from-scratch rebuild records, per family of tables.

Motivation: the manuscript claimed "reproduced to within 0.33". That bound appeared in no
released record. This script derives the bound from the shipped JSONs so the number in the
paper is traceable. Pure file IO, no models, no GPU.

Run: python3 scripts/repro_deviation.py          # writes results/extended/repro_deviation.json
"""
from __future__ import annotations

import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(p):
    with open(os.path.join(ROOT, p)) as fh:
        return json.load(fh)


def flatten(obj, prefix=""):
    """Yield (dotted-key, float) for every FINITE numeric leaf (NaN/inf are skipped and
    reported separately, never silently treated as agreement)."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if math.isfinite(float(obj)):
            out[prefix] = float(obj)
    return out


def nonfinite_keys(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += nonfinite_keys(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if not math.isfinite(float(obj)):
            out.append(prefix)
    return out


def compare(name, a_path, b_path, subkey=None, note=""):
    A, B = load(a_path), load(b_path)
    if subkey:
        A, B = A[subkey], B[subkey]
    fa, fb = flatten(A), flatten(B)
    shared = sorted(set(fa) & set(fb))
    diffs = [(abs(fa[k] - fb[k]), k, fa[k], fb[k]) for k in shared]
    diffs.sort(reverse=True)
    top = diffs[0] if diffs else (0.0, None, None, None)
    # values that actually appear as a printed table entry are the ".mean" leaves
    md = [d for d in diffs if d[1].endswith(".mean") or "." not in d[1]]
    mtop = md[0] if md else (0.0, None, None, None)
    return {
        "family": name,
        "published_record": a_path,
        "rebuild_record": b_path,
        "n_values_compared": len(shared),
        "max_abs_deviation": round(top[0], 4),
        "max_at": top[1],
        "published_value": top[2],
        "rebuild_value": top[3],
        "max_abs_deviation_printed_means_only": round(mtop[0], 4),
        "max_at_printed_means_only": mtop[1],
        "top5": [{"key": k, "published": p, "rebuild": r, "abs_dev": round(d, 4)}
                 for d, k, p, r in diffs[:5]],
        "not_comparable_nonfinite": sorted(set(nonfinite_keys(A)) | set(nonfinite_keys(B))),
        "note": note,
    }


def main():
    rows = []
    # --- dense-CLIP regime, ADE20K-150: the record that backs tab_real_regime (results/extended)
    rows.append(compare(
        "dense-CLIP ADE20K (tab_real_regime)",
        "results/extended/metrics_regime.json", "results/rebuild/metrics_regime.json",
        subkey="summary",
        note="Deviation is confined to the TRAINED decoder row (stochastic 12-epoch training); "
             "every parameter-free row is bit-identical."))
    # --- dense-CLIP regime, PASCAL VOC: backs tab_real_voc (transcribed from results/metrics_voc.json)
    rows.append(compare(
        "dense-CLIP VOC2012 (tab_real_voc)",
        "results/metrics_voc.json", "results/rebuild/metrics_voc.json",
        subkey="summary",
        note="Deviation is confined to the TRAINED decoder row."))
    # --- class-symmetric / AURC baselines: back tab_real_baselines (results/extended)
    for f in ("metrics_stage3.json", "metrics_stage4.json"):
        rows.append(compare(
            f"dense-CLIP baselines ({f})",
            f"results/extended/{f}", f"results/rebuild/{f}", subkey="summary",
            note="The rebuild records AURC as NaN for the two trained-decoder rows "
                 "(risk-coverage curve degenerate under the rebuild's confidence ties); those "
                 "two values are excluded from the bound rather than counted as agreement."))
    rows.append(compare(
        "adaptive/class-balanced ECE (metrics_adaece.json)",
        "results/metrics_adaece.json", "results/rebuild/metrics_adaece.json"))
    # --- masked-crop regime: rebuild vs rebuild (both on the 1623-image rebuilt cache)
    for f in ("metrics_protomask.json", "metrics_diffuse.json", "metrics_calib.json"):
        rows.append(compare(
            f"masked-crop rebuilt cache ({f})",
            f"results/extended/mc_{f}", f"results/rebuild_mc/{f}",
            subkey="summary" if f == "metrics_calib.json" else None,
            note="Both records are on the SAME rebuilt 1623-image cache; this is run-to-run "
                 "variance of the trained decoder / diffusion sweep, not cache-to-cache."))

    dense = [r for r in rows if r["family"].startswith("dense-CLIP")
             or r["family"].startswith("adaptive")]
    mc = [r for r in rows if r["family"].startswith("masked-crop")]
    out = {
        "_comment": "Max absolute deviation between published result records and the independent "
                    "from-scratch rebuild. Generated by scripts/repro_deviation.py. All values are "
                    "in the units of the metric as reported (IoU/ECE points on a 0-100 scale).",
        "dense_regime_max_abs_deviation": round(max(r["max_abs_deviation"] for r in dense), 4),
        "dense_regime_max_at": max(dense, key=lambda r: r["max_abs_deviation"])["max_at"],
        "dense_regime_max_abs_deviation_printed_means_only": round(
            max(r["max_abs_deviation_printed_means_only"] for r in dense), 4),
        "dense_regime_max_at_printed_means_only": max(
            dense, key=lambda r: r["max_abs_deviation_printed_means_only"]
        )["max_at_printed_means_only"],
        "dense_regime_parameter_free_rows_max_abs_deviation": 0.0,
        "masked_crop_rebuilt_cache_max_abs_deviation": round(
            max(r["max_abs_deviation"] for r in mc), 4),
        "families": rows,
    }
    # verify the "parameter-free rows are identical" sub-claim rather than asserting it
    pf_max = 0.0
    for pub, reb in (("results/extended/metrics_regime.json", "results/rebuild/metrics_regime.json"),
                     ("results/metrics_voc.json", "results/rebuild/metrics_voc.json")):
        A, B = load(pub)["summary"], load(reb)["summary"]
        for row in ("uncal", "globalT", "perclassT", "diffusion"):
            fa, fb = flatten(A[row]), flatten(B[row])
            for k in set(fa) & set(fb):
                pf_max = max(pf_max, abs(fa[k] - fb[k]))
    out["dense_regime_parameter_free_rows_max_abs_deviation"] = round(pf_max, 4)

    dst = os.path.join(ROOT, "results", "extended", "repro_deviation.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {dst}")
    for r in rows:
        print(f"  {r['family']:48s} max|d|={r['max_abs_deviation']:.2f} at {r['max_at']}"
              f"  ({r['published_value']} -> {r['rebuild_value']})")
    print(f"\ndense regime max |deviation|          = {out['dense_regime_max_abs_deviation']}")
    print(f"dense regime parameter-free rows max = "
          f"{out['dense_regime_parameter_free_rows_max_abs_deviation']}")
    print(f"masked-crop rebuilt-cache max        = "
          f"{out['masked_crop_rebuilt_cache_max_abs_deviation']}")


if __name__ == "__main__":
    main()
