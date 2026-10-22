# Frozen models and external code

Every model below is used **frozen** (no fine-tuning). The fitted components are the correction heads, calibrators, and learned graph affinity.

## Hugging Face models (masked-crop pipeline, `src/protomask/core.py`)

| Role | Identifier | Loader |
|---|---|---|
| Mask proposer | `facebook/sam-vit-base` | `transformers.pipeline("mask-generation")` |
| Region + text encoder | `openai/clip-vit-large-patch14` | `transformers.CLIPModel` / `CLIPProcessor` |

These are downloaded from the Hugging Face Hub on first use. The reported run used these exact
Hub revisions (pass them to `from_pretrained(..., revision=...)` for byte-exact reproduction):

| Identifier | Revision (commit) |
|---|---|
| `facebook/sam-vit-base` | `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f` |
| `openai/clip-vit-large-patch14` | `32bd64288804d66eefd0ccbe215aa642df71cc41` |

## Dense-CLIP regime (competitive backbone, Tier C)

The competitive training-free segmenter is a SCLIP-style Correlative Self-Attention (CSA) dense CLIP
built on the **OpenAI CLIP `ViT-B/16`** weights:

Use the modified `clip` module supplied by the SCLIP checkout, which implements
`encode_image(..., return_all=True, csa=True)`. A separately installed OpenAI `clip` does not
provide those arguments. Scripts prioritize the checkout and check where `clip` was imported.

```bash
git clone https://github.com/wangf3014/SCLIP.git SCLIP
git -C SCLIP checkout b56bd3014b9e8a3ee3450bcfc53700a67d627b7a
export PMX_SCLIP=SCLIP
pip install ftfy regex
```

The weights are OpenAI CLIP ViT-B/16; the attention implementation is SCLIP's. LPOSS and the
proxy base also use DINO ViT-B/16 through torch.hub. The historical DINO source commit was
not recorded; no exact revision is claimed for those runs.

## Datasets

* **ADE20K-150** (`ADEChallengeData2016`) — point `PMX_ADE_ROOT` at the extracted root.
* **PASCAL VOC2012** (`VOCdevkit/VOC2012`) — point `PMX_VOC_ROOT` at the extracted root.

Both are standard public benchmarks; we use the official `validation` / `val.txt` splits and the
shipped seen/novel class splits under `splits/`.

COCO-Stuff-171 transfer data and paths are described in `data/README.md`.
