# SegFormer-RDA v1.0.0

## Contents

- SegFormer-RDA source code
- Training, inference, and mIoU evaluation scripts
- Environment and dataset layout documentation
- Git LFS objects containing the trained checkpoints and original source data

## Model

This release uses a B2 SegFormer backbone and a two-class segmentation head.
The modified decoder includes RASPP-based multi-scale context enhancement and
decoupled channel attention.

## Release assets

- `segformer-rda-weights-v1.0.0.zip` (optional mirror)
- `segformer-rda-dataset-v1.0.0.zip` (optional mirror)

## Dataset status

The source material contains 54 full-size image/mask pairs, but not the
1,147/203 patch-level split required by the training scripts. The repository
preserves this source snapshot exactly under `liefeng/`; it must not be
described as the canonical reproducible patch dataset.
