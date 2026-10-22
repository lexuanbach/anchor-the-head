"""Persist an auditable provenance record for the LPOSS propagation we run verbatim.

`scripts/lposs_extract.py` only PRINTS the per-function SHA-256 at extraction time, so a reader of
the artifact had no way to check the "verbatim / SHA-verified" claim. This script writes the
digests to `results/extended/lposs_provenance.json` and, given an upstream checkout (or a downloaded
copy of the single source file), verifies them against the authors' released code.

Digest definition (identical to lposs_extract.py): for each function, take the source lines
`[node.lineno-1 : node.end_lineno]` of the file that defines it, join them with line endings kept,
and SHA-256 the resulting bytes. `sha256_rstrip` is the same text with trailing whitespace stripped;
it is the digest that survives our extractor's `.rstrip()` when it writes lposs_core_auto.py, so it
is the one that can be compared across the upstream file and our extracted module.

Usage:
  python3 scripts/lposs_provenance.py                               # digest our extracted module
  python3 scripts/lposs_provenance.py --upstream-file /tmp/lposs_eval.py
  python3 scripts/lposs_provenance.py --lposs /path/to/LPOSS        # upstream checkout
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUNCS = ["normalize_connection_graph", "get_lposs_laplacian", "dfs_search", "perform_lp"]

UPSTREAM = {
    "repo": "https://github.com/vladan-stojnic/LPOSS",
    "paper": "Stojnic, Kalantidis, Matas, Tolias. LPOSS: Label Propagation over Patches and "
             "Pixels for Open-vocabulary Semantic Segmentation. CVPR 2025.",
    "file": "segmentation/evaluation/lposs_eval.py",
}


def func_digests(path):
    src = open(path).read()
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in FUNCS:
            body = "".join(lines[node.lineno - 1:node.end_lineno])
            out[node.name] = {
                "n_source_lines": node.end_lineno - node.lineno + 1,
                "sha256": hashlib.sha256(body.encode()).hexdigest(),
                "sha256_rstrip": hashlib.sha256(body.rstrip().encode()).hexdigest(),
            }
    missing = [f for f in FUNCS if f not in out]
    if missing:
        raise SystemExit(f"ERROR: functions not found in {path}: {missing}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default=os.path.join(ROOT, "scripts", "lposs_core_auto.py"),
                    help="our extracted module (the code actually executed by pm_stage8_lposs.py)")
    ap.add_argument("--lposs", default=None, help="path to an upstream LPOSS checkout")
    ap.add_argument("--upstream-file", default=None,
                    help="path to a copy of segmentation/evaluation/lposs_eval.py")
    ap.add_argument("--upstream-commit", default=None,
                    help="commit the upstream file was taken from, recorded verbatim")
    ap.add_argument("--upstream-source", default=None,
                    help="how the upstream copy was obtained (URL or checkout), recorded verbatim")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "extended", "lposs_provenance.json"))
    a = ap.parse_args()

    ours = func_digests(a.module)
    rec = {
        "_comment": "Provenance of the LPOSS propagation executed by scripts/pm_stage8_lposs.py. "
                    "Regenerate with scripts/lposs_provenance.py; verify against upstream with "
                    "--lposs /path/to/LPOSS.",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "upstream": dict(UPSTREAM),
        "extracted_module": os.path.relpath(a.module, ROOT),
        "extracted_module_sha256": hashlib.sha256(open(a.module, "rb").read()).hexdigest(),
        "digest_definition": "SHA-256 over the function's source lines as written; "
                             "sha256_rstrip is the same text with trailing whitespace stripped "
                             "(the form our extractor emits, and the form to compare across files).",
        "functions": ours,
        "verified_against_upstream": False,
        "documented_deviations": [
            {
                "what": "faiss.knn_gpu -> exact torch top-k",
                "where": "scripts/pm_stage8_lposs.py (shim applied after import; the extracted "
                         "module is never edited)",
                "why": "prebuilt faiss GPU wheels ship no sm_90 kernels, so faiss.knn_gpu raises "
                       "CUDA error 209 on an H100; faiss-cpu is used for the import only.",
                "effect": "the shim returns the exact same k nearest neighbours by cosine "
                          "similarity (torch.topk on the full similarity matrix), i.e. an exact "
                          "rather than approximate kNN; it is not an algorithmic change to the "
                          "propagation.",
            },
            {
                "what": "logits and affinities come from this paper's cached dense-CLIP (SCLIP-CSA) "
                        "stack and our five seen/novel splits, not from LPOSS's own pipeline",
                "where": "scripts/pm_stage8_lposs.py",
                "why": "the comparison isolates the propagation mechanism under a matched protocol.",
                "effect": "the reported 15.1 mIoU is NOT LPOSS's published ADE20K number and no "
                          "claim is made about LPOSS's own full-resolution system.",
            },
        ],
    }

    up_path = a.upstream_file
    if up_path is None and a.lposs:
        up_path = os.path.join(a.lposs, "segmentation", "evaluation", "lposs_eval.py")
    if up_path:
        up = func_digests(up_path)
        match = {f: up[f]["sha256_rstrip"] == ours[f]["sha256_rstrip"] for f in FUNCS}
        rec["upstream"]["retrieved_from"] = a.upstream_source or up_path
        rec["upstream"]["upstream_file_sha256"] = hashlib.sha256(
            open(up_path, "rb").read()).hexdigest()
        if a.upstream_commit:
            rec["upstream"]["commit"] = a.upstream_commit
        rec["upstream_functions"] = up
        rec["per_function_match"] = match
        rec["verified_against_upstream"] = all(match.values())
        # the raw (non-rstripped) digests match too when the upstream file is byte-identical
        rec["per_function_match_raw"] = {
            f: up[f]["sha256"] == ours[f]["sha256"] for f in FUNCS}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"wrote {a.out}")
    for f in FUNCS:
        line = f"  {f:28s} sha256={ours[f]['sha256'][:16]}"
        if "per_function_match" in rec:
            line += f"  upstream_match={rec['per_function_match'][f]}"
        print(line)
    print(f"  verified_against_upstream = {rec['verified_against_upstream']}")


if __name__ == "__main__":
    main()
