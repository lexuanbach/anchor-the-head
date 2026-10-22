# Frozen models and external code

Every model below is used **frozen** (no fine-tuning). The only trained component anywhere in the
paper is the small graph-decoder / calibration head, which is the audited negative result.

## Hugging Face models (masked-crop pipeline, `src/protomask/core.py`)

| Role | Identifier | Loader |
|---|---|---|
| Mask proposer | `facebook/sam-vit-base` | `transformers.pipeline("mask-generation")` |
| Region + text encoder | `openai/clip-vit-large-patch14` | `transformers.CLIPModel` / `CLIPProcessor` |

These are downloaded from the Hugging Face Hub on first use. The submitted run used these exact
Hub revisions (pass them to `from_pretrained(..., revision=...)` for byte-exact reproduction):

| Identifier | Revision (commit) |
|---|---|
| `facebook/sam-vit-base` | `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f` |
| `openai/clip-vit-large-patch14` | `32bd64288804d66eefd0ccbe215aa642df71cc41` |

## Dense-CLIP regime (competitive backbone, Tier C)

The competitive training-free segmenter is a SCLIP-style class-symmetric-attention (CSA) dense CLIP
built on the **OpenAI CLIP `ViT-B/16`** weights:

* OpenAI CLIP package: `pip install git+https://github.com/openai/CLIP.git` (provides `clip`, loads `ViT-B/16`).
* SCLIP repository (CSA attention, prompt templates, class-name configs): clone it and point
  `PMX_SCLIP` at the checkout. The submitted run used SCLIP commit
  `b56bd3014b9e8a3ee3450bcfc53700a67d627b7a`. Used by `pm_stage1.py`, `pm_stage2.py`, `sclip_eval.py`, `voc_collapse.py`.

## Datasets

* **ADE20K-150** (`ADEChallengeData2016`) — point `PMX_ADE_ROOT` at the extracted root.
* **PASCAL VOC2012** (`VOCdevkit/VOC2012`) — point `PMX_VOC_ROOT` at the extracted root.

Both are standard public benchmarks; we use the official `validation` / `val.txt` splits and the
shipped seen/novel class splits under `splits/`.
