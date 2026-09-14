# Notices and Attribution

SegFormer-RDA contains code derived from the SegFormer PyTorch implementation:

<https://github.com/bubbliiiing/segformer-pytorch>

The upstream implementation was used as the baseline for the SegFormer
backbone, training loop, inference pipeline, dataset loader, and evaluation
utilities. This repository modifies the decoder architecture and training
configuration for two-class coal-rock fracture segmentation.

The files under `nets/` retain the NVIDIA copyright and NVIDIA Source Code
License headers present in the source material. Those notices must remain with
the corresponding files. Review the upstream repository and its license before
redistributing a fork or selecting a repository-wide license for this project.

The fracture dataset and trained checkpoints are separate release assets. Their
ownership and redistribution terms must be confirmed by the dataset/model
owner before publishing them publicly.

