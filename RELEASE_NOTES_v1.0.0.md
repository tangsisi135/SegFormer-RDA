# SegFormer-RDA v1.0.0

## Contents

- SegFormer-RDA source code
- Training, inference, and mIoU evaluation scripts
- Environment and dataset layout documentation
- Release assets containing the trained checkpoint and validated dataset, when
  their redistribution terms have been confirmed

## Model

This release uses a B2 SegFormer backbone and a two-class segmentation head.
The modified decoder includes RASPP-based multi-scale context enhancement and
decoupled channel attention.

## Release assets

- `segformer-rda-weights-v1.0.0.zip`
- `segformer-rda-dataset-v1.0.0.zip` (pending canonical patch dataset and
  split-file validation)

## Dataset status

The source material inspected while preparing this release contains 54
full-size image/mask pairs, but not the 1,147/203 patch-level split required by
the training scripts. It must not be labeled as the reproducible training
dataset until the patch dataset and `train.txt`/`val.txt` files are supplied.
