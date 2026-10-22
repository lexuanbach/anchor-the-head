"""Regenerate the headline real tables (tab_real_main, tab_real_diffuse) from the full-val JSONs so
every real table reports the true number of evaluated val images (cache_features.py skips images
where SAM returns <2 masks, so this is NOT always 2000). Reads results/metrics_diffuse.json
(diffusion sweep, seed-0 split) and results/metrics_protomask.json (trained decoder) if present.
Pure file IO; no models. Run: python scripts/finalize_tables.py --out-dir results
"""
from __future__ import annotations

import argparse
import json
import os


CAPTION_OVERRIDES = {
 'tab_real_main.tex': (
  '% AUTO-GENERATED (REAL) by scripts/finalize_tables.py -- ADE20K-150 val, ORIGINAL masked-crop cache: every SAM-cacheable image of the 2000-image val split (the exact cached count was not recorded for this cache), 256px, frozen SAM(ViT-B)+CLIP(ViT-L/14), no fine-tuning.',
  'Masked-crop comparison on ADE20K-150 val, frozen SAM$+$CLIP, original cache, seed-$0$ split. hIoU is the seen/novel harmonic mean. Lower ECE is better.'),
 'tab_real_diffuse.tex': (
  '% AUTO-GENERATED (REAL) by scripts/finalize_tables.py -- ADE20K-150 val, ORIGINAL masked-crop cache: every SAM-cacheable image of the 2000-image val split.',
  'Diffusion sweep over weight $\\alpha$ and steps $k$. ADE20K-150 val, original masked-crop cache. The $\\alpha{=}0$ row is the independent Proposal$+$CLIP baseline.'),
 'tab_real_decoder.tex': (
  '% AUTO-GENERATED (REAL) by scripts/finalize_tables.py -- ADE20K-150 val, ORIGINAL masked-crop cache: every SAM-cacheable image of the 2000-image val split, trained decoder (12 epochs).',
  'Decoder ablation on ADE20K-150 val, original masked-crop cache. Rows remove the graph, calibration, or prototype-refinement component of the trained decoder.'),
}


# Per-table layout tweaks used by the manuscript (kept here so regeneration is byte-exact).
LAYOUT = {
    "tab_real_main.tex": {"after_label": "\\small\\setlength{\\tabcolsep}{2.2pt}"},
    "tab_real_decoder.tex": {"centering": "\\centering",
                             "after_label": "\\setlength{\\tabcolsep}{10pt}",
                             "tabular": "\\begin{tabular}{@{}lccccc@{}}"},
}


def _emit(path, lines, use_override=True):
    """Write a table, applying the manuscript caption/provenance override if one exists.

    The manuscript uses shorter captions than the verbose generated ones. They live HERE so the
    generating script stays the single source of truth AND so regeneration is byte-identical to the
    shipped paper/ copy (verify.py checks exactly that). The overrides quote numbers from the
    ORIGINAL masked-crop cache, so they are suppressed (use_override=False) when the script is run
    over a different cache, e.g. the independent rebuild.
    """
    import re as _re
    text = "\n".join(lines) + "\n"
    ov = CAPTION_OVERRIDES.get(os.path.basename(path)) if use_override else None
    if ov:
        hdr, cap = ov
        rows = text.split("\n")
        if rows and rows[0].startswith("%"):
            rows[0] = hdr
            text = "\n".join(rows)
        m = _re.search(r"\\caption\{", text)
        if m:
            i = m.end(); d = 1; j = i
            while d and j < len(text):
                d += (text[j] == "{") - (text[j] == "}"); j += 1
            text = text[:m.start()] + "\\caption{" + cap + "}" + text[j:]
    lay = LAYOUT.get(os.path.basename(path)) if use_override else None
    if lay:
        rows = text.rstrip("\n").split("\n")
        out = []
        for r in rows:
            if r.startswith("\\centering") and "centering" in lay:
                r = lay["centering"]
            if r.startswith("\\begin{tabular}") and "tabular" in lay:
                r = lay["tabular"]
            out.append(r)
            if r.startswith("\\label{") and "after_label" in lay:
                out.append(lay["after_label"])
        text = "\n".join(out) + "\n"
    with open(path, "w") as fh:
        fh.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-val", type=int, default=None,
                    help="cached-image count for THIS cache, when the headline JSON does not record "
                         "it (e.g. read from the same cache's metrics_calib.json config.n_val)")
    a = ap.parse_args()
    R = a.out_dir
    diff = json.load(open(os.path.join(R, "metrics_diffuse.json")))
    base = diff["baseline (alpha=0)"]
    head = diff.get("alpha=0.4,k=2", base)  # fixed-a-priori headline diffusion setting
    dec = None
    pmp = os.path.join(R, "metrics_protomask.json")
    if os.path.exists(pmp):
        pm = json.load(open(pmp))
        dec = pm.get("full")

    # The evaluated set is the number of CACHED images, not --n-images: cache_features.py skips
    # any image where SAM returns <2 masks, so it is NOT 2000. The count must come from the
    # headline record itself; results/metrics_calib.json describes the SEPARATE 347-image
    # five-split protocol and must not be used as the headline count.
    NVAL = diff.get("config", {}).get("n_val") or a.n_val
    nstr = (f"N={NVAL} SAM-cacheable images of the 2000-image val split" if NVAL else
            "N=every SAM-cacheable image of the 2000-image val split; the exact cached count "
            "was not recorded for this cache")
    # The LaTeX form used inside captions.
    ncap = (f"$N{{=}}{NVAL}$ SAM-cacheable images of the $2000$-image val split" if NVAL else
            "every SAM-cacheable image of the $2000$-image val split")

    os.makedirs(os.path.join(R, "tables"), exist_ok=True)

    # ---- tab_real_main : baseline vs headline diffusion vs trained decoder ----
    L = [f"% AUTO-GENERATED (REAL) by scripts/finalize_tables.py -- ADE20K-150 val ({nstr}, 256px), "
         "frozen SAM(ViT-B)+CLIP(ViT-L/14), no fine-tuning.",
         "\\centering\\small",
         f"\\caption{{\\textbf{{Real}} open-vocabulary segmentation on ADE20K-150 val "
         f"({ncap}, frozen SAM masks $+$ frozen CLIP, no fine-tuning; seed-0 split). "
         "Parameter-free graph \\emph{diffusion} of the frozen-CLIP per-mask logits is on par with the "
         "independent-scoring baseline on mIoU, improves novel-class (open-vocab) IoU, and \\emph{halves} "
         "calibration error; the \\emph{trained} graph decoder overfits seen classes and collapses "
         "novel-class IoU. The absolute mIoU is low because the region encoding is a simple masked-crop "
         "of frozen CLIP; the contribution is the mechanism, not SOTA accuracy.}",
         "\\label{tab:real_main}",
         "\\begin{tabular}{lcccc}", "\\toprule",
         "Method & mIoU & hIoU & novel-IoU & ECE$\\downarrow$ \\\\", "\\midrule",
         f"Proposal+CLIP (independent) & {base['mIoU']} & {base['hIoU']} & {base['novel']} & {base['ECE']} \\\\",
         f"\\textbf{{Graph diffusion}} ($\\alpha{{=}}0.4,k{{=}}2$) & {head['mIoU']} & {head['hIoU']} & "
         f"\\textbf{{{head['novel']}}} & \\textbf{{{head['ECE']}}} \\\\"]
    if dec is not None:
        L.append(f"Trained graph decoder & {dec['mIoU']} & {dec['hIoU']} & {dec['novelIoU']} & {dec['ECE']} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    _emit(os.path.join(R, "tables", "tab_real_main.tex"), L, use_override=(a.n_val is None))

    # ---- tab_real_diffuse : full alpha x k sweep ----
    order = ["alpha=0.2,k=1", "alpha=0.2,k=2", "alpha=0.2,k=3",
             "alpha=0.4,k=1", "alpha=0.4,k=2", "alpha=0.4,k=3",
             "alpha=0.6,k=1", "alpha=0.6,k=2", "alpha=0.6,k=3",
             "alpha=0.8,k=1", "alpha=0.8,k=2", "alpha=0.8,k=3"]
    # find best mIoU and best (lowest) ECE for bolding
    best_miou = max(diff[k]["mIoU"] for k in order if k in diff)
    best_ece = min(diff[k]["ECE"] for k in order if k in diff)
    best_novel = max(diff[k]["novel"] for k in order if k in diff)
    best_hiou = max(diff[k]["hIoU"] for k in order if k in diff)

    def fmt(k):
        m = diff[k]
        a, kk = k.split(",")
        a = a.split("=")[1]; kk = kk.split("=")[1]
        miou = f"\\textbf{{{m['mIoU']}}}" if m["mIoU"] == best_miou else f"{m['mIoU']}"
        ece = f"\\textbf{{{m['ECE']}}}" if m["ECE"] == best_ece else f"{m['ECE']}"
        nov = f"\\textbf{{{m['novel']}}}" if m["novel"] == best_novel else f"{m['novel']}"
        hio = f"\\textbf{{{m['hIoU']}}}" if m["hIoU"] == best_hiou else f"{m['hIoU']}"
        return f"  $\\alpha{{=}}{a},k{{=}}{kk}$ & {miou} & {hio} & {nov} & {ece} \\\\"

    D = [f"% AUTO-GENERATED (REAL) by scripts/finalize_tables.py -- ADE20K-150 val ({nstr}).",
         "\\centering\\small",
         f"\\caption{{\\textbf{{Real}} graph-diffusion sweep over diffusion weight $\\alpha$ and steps $k$ "
         f"(ADE20K-150 val, {ncap}, seed-0 split). $\\alpha{{=}}0$ is the independent "
         "Proposal+CLIP baseline. Diffusion is parameter-free and symmetric over all classes, so it "
         "cannot collapse novel classes; it trades a little mIoU for a large calibration gain.}",
         "\\label{tab:real_diffuse}",
         "\\begin{tabular}{lcccc}", "\\toprule",
         "Config & mIoU & hIoU & novel-IoU & ECE$\\downarrow$ \\\\", "\\midrule",
         f"  baseline ($\\alpha{{=}}0$) & {base['mIoU']} & {base['hIoU']} & {base['novel']} & {base['ECE']} \\\\",
         "\\midrule"]
    D += [fmt(k) for k in order if k in diff]
    D += ["\\bottomrule", "\\end{tabular}"]
    _emit(os.path.join(R, "tables", "tab_real_diffuse.tex"), D, use_override=(a.n_val is None))

    # ---- tab_real_decoder : trained-decoder negative result ----
    if os.path.exists(pmp):
        pm = json.load(open(pmp))
        b = pm["baseline"]
        rows = [("full (graph+residual+calib+proto)", "full"),
                ("$-$ graph (per-node MLP)", "no_graph"),
                ("$-$ calibration head", "no_uncertainty"),
                ("$-$ prototype refinement", "fixed_protos")]
        T = [f"% AUTO-GENERATED (REAL) by scripts/finalize_tables.py -- ADE20K-150 val ({nstr}), "
             "trained decoder (12 epochs).",
             "\\centering\\small",
             f"\\caption{{\\textbf{{Real}} negative result for the \\emph{{trained}} graph decoder "
             f"(ADE20K-150 val, {ncap}; $120$ seen / $30$ held-out novel classes, seed-0 split). The "
             "graph variants match the baseline on \\emph{seen}-class IoU but collapse \\emph{novel}-class "
             "IoU---the open-vocabulary failure of a head supervised only on seen classes; removing "
             "prototype refinement collapses \\emph{both}. This is why the validated mechanism "
             "(Table~\\ref{tab:real_main}) is parameter-free diffusion, not a learned decoder.}",
             "\\label{tab:real_decoder}",
             "\\begin{tabular}{lccccc}", "\\toprule",
             "Variant & mIoU & seen-IoU & novel-IoU & hIoU & ECE$\\downarrow$ \\\\", "\\midrule",
             f"  Proposal+CLIP baseline & {b['mIoU']} & {b['seenIoU']} & {b['novelIoU']} & {b['hIoU']} & {b['ECE']} \\\\",
             "\\midrule"]
        for label, key in rows:
            if key in pm:
                m = pm[key]
                T.append(f"  {label} & {m['mIoU']} & {m['seenIoU']} & {m['novelIoU']} & {m['hIoU']} & {m['ECE']} \\\\")
        T += ["\\bottomrule", "\\end{tabular}"]
        _emit(os.path.join(R, "tables", "tab_real_decoder.tex"), T, use_override=(a.n_val is None))
        print(f"[finalize] wrote tab_real_decoder.tex ({nstr})", flush=True)
    print(f"[finalize] wrote tab_real_main.tex + tab_real_diffuse.tex ({nstr})", flush=True)



if __name__ == "__main__":
    main()
