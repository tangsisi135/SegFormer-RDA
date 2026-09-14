# Model weights

Model checkpoints are intentionally not stored in the Git repository. Download
the trained checkpoint from the GitHub Release and place it here as:

```text
weights/best_epoch_weights.pth
```

Alternatively, set `SEGFORMER_RDA_WEIGHTS` for inference or
`SEGFORMER_RDA_INIT_WEIGHTS` for training. The checkpoint must match the
two-class `SegFormer-RDA` B2 model used by this repository.

The optional ImageNet-pretrained SegFormer B2 backbone weights are separate
from the trained segmentation checkpoint. To use them when constructing a
model with `pretrained=True`, put the upstream file in `model_data/` or set
`SEGFORMER_RDA_BACKBONE_DIR` to the directory containing it.

