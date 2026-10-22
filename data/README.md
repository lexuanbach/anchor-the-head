# Data and evaluation protocols

Backbones remain frozen; the training partitions below supply labels for fitting correction heads
and calibrators. Dataset images and feature caches are not included in the repository.

## ADE20K-150

Obtain `ADEChallengeData2016` from the [ADE20K project](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
and extract it under `data/`, or set `PMX_ADE_ROOT` to its root:

```text
ADEChallengeData2016/images/{training,validation}/*.jpg
ADEChallengeData2016/annotations/{training,validation}/*.png
```

Annotation 0 is ignored; labels 1-150 map to classes 0-149. Dense CLIP evaluates all 2000
validation images and fits on 600 training images. Masked-crop accuracy evaluates the subset
for which SAM supplies at least two proposals (original count unrecorded, rebuild 1623).
The small calibration/encoder record has 347 validation and 300 training images; its edge
ablation uses seed 0 only. Masked-crop class splits and dense class splits use different RNG
protocols, documented in `../REPRODUCE.md` and shipped separately in `../splits/`.

## PASCAL VOC2012

Download the train/validation release from the [VOC2012 project](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
Extract `VOCdevkit/VOC2012` under `data/`, or set `PMX_VOC_ROOT`:

```text
VOCdevkit/VOC2012/JPEGImages/*.jpg
VOCdevkit/VOC2012/SegmentationClass/*.png
VOCdevkit/VOC2012/ImageSets/Segmentation/{train,val}.txt
```

The study scores 20 object classes, with six novel per split. Background (0) and void (255)
are excluded from the object-class metrics. Corrections fit on seen-class training patches.

## COCO-Stuff-171 transfer

From the [official COCO-Stuff downloads](https://github.com/nightrome/cocostuff#downloads),
download `val2017.zip` and the stuff+thing PNG annotations, `stuffthingmaps_trainval2017.zip`.
The transfer script fits on 600 COCO val2017 images and evaluates on ADE20K without fitting ADE
labels. COCO and ADE class names can overlap.

```text
data/coco/val2017/*.jpg
data/coco/stuffthingmaps/val2017/*.png
```

Set `PMX_COCO_IMG` and `PMX_COCO_ANN` to these directories. Set `PMX_ADE_NAMES` to the SCLIP
`configs/cls_coco_stuff.txt` when running `pm_stage9a_cache_coco.py`. See `../REPRODUCE.md` for
commands and separate cache locations, and `../MODELS.md` for frozen models.
