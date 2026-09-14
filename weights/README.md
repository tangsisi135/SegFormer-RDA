# Model weights

The repository preserves the original checkpoint locations. The trained
SegFormer-RDA checkpoints are in `logs_improve/`, while the optional upstream
backbone checkpoints are in `model_data/`:

```text
logs_improve/best_epoch_weights.pth
logs_improve/ep100-loss0.011-val_loss0.012.pth
logs_improve/ep105-loss0.011-val_loss0.012.pth
model_data/segformer_b0_weights_voc.pth
model_data/segformer_b1_weights_voc.pth
model_data/segformer_b2_weights_voc.pth
```

The inference scripts default to `logs_improve/best_epoch_weights.pth`. To use
a different location, set `SEGFORMER_RDA_WEIGHTS`. The checkpoint must match the two-class
`SegFormer-RDA` B2 model used by this repository.

The optional ImageNet-pretrained SegFormer B2 backbone weights are separate
from the trained segmentation checkpoint. To use them when constructing a
model with `pretrained=True`, put the upstream file in `model_data/` or set
`SEGFORMER_RDA_BACKBONE_DIR` to the directory containing it.

All `.pth` files are tracked with Git LFS. Run `git lfs pull` after cloning to
download the actual checkpoint files.
