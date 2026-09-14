# SegFormer-RDA

SegFormer-RDA is a PyTorch semantic segmentation project for two-class coal
rock fracture segmentation. It uses a SegFormer-B2 backbone and a modified
decoder with RASPP multi-scale context enhancement and decoupled channel
attention.

## Repository contents

This repository follows the original experiment-directory architecture and
contains the source code, dataset, trained checkpoints, generated results,
training logs, and Python cache files. The large `.pth` and `.tif` files are
stored with Git LFS so their paths remain unchanged while ordinary Git stores
only small pointer files.

## Attribution

The project is derived from the SegFormer PyTorch implementation by
[bubbliiiing](https://github.com/bubbliiiing/segformer-pytorch). The upstream
implementation provided the baseline backbone, training loop, inference
pipeline, dataset loader, and evaluation utilities. This repository modifies
the decoder architecture and training configuration for fracture segmentation.

The files under `nets/` retain the NVIDIA copyright and NVIDIA Source Code
License headers from the source material. See [NOTICE.md](NOTICE.md) before
redistributing this project. Do not remove the upstream notices or assume that
the upstream code is covered by a different license without checking its terms.

## Environment

The recorded experiment environment was:

- Windows 11 x64
- Python 3.11
- PyTorch 2.7.0
- NVIDIA GeForce RTX 4080 SUPER 16 GB
- CUDA 12.6 PyTorch build recommended for the recorded setup

The AMD Threadripper PRO CPU is suitable for data loading. CUDA is optional for
inference and development, but training the B2 model is expected to require a
CUDA-capable NVIDIA GPU.

### pip installation

Create and activate a clean Python 3.11 environment, then install the
PyTorch build matching the installed NVIDIA driver. For the recorded CUDA 12.6
setup:

```powershell
python -m pip install -r requirements-cuda126.txt
python -m pip install -r requirements.txt
python check_environment.py
```

If CUDA 12.6 wheels are not appropriate for the installed driver, follow the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/)
and then install `requirements.txt`.

An alternative Conda environment definition is provided in
`environment-windows.yml`.

The large-image script `predictbig.py` additionally requires GDAL. Conda users
can install it with:

```powershell
conda install -c conda-forge gdal
```

GDAL is intentionally not imported by the ordinary training and image
inference scripts.

## Dataset

The dataset is stored in the repository under `liefeng/` and follows this
layout:

```text
liefeng/
└── VOC2007/
    ├── JPEGImages/<patch-id>.tif
    ├── SegmentationClass/<patch-id>.tif
    └── ImageSets/Segmentation/
        ├── train.txt
        └── val.txt
```

The canonical patch-level split contains 1,147 training patches and 203
validation patches. Check it with:

```powershell
python prepare_dataset_splits.py
```

To keep the dataset elsewhere, set the root directory containing `VOC2007`:

```powershell
$env:SEGFORMER_RDA_DATASET_ROOT = 'D:\datasets\liefeng'
```

See [data/README.md](data/README.md) for the full layout.

The uploaded source folder contains 54 full-size image/mask pairs. It does not
contain the 1,147/203 patch-level split files used by the training script, so
the included data should be treated as the original experiment data snapshot,
not as a verified reproducible patch split.

## Model weights

The original checkpoints remain in their source locations:

```text
logs_improve/best_epoch_weights.pth
logs_improve/ep100-loss0.011-val_loss0.012.pth
logs_improve/ep105-loss0.011-val_loss0.012.pth
model_data/segformer_b0_weights_voc.pth
model_data/segformer_b1_weights_voc.pth
model_data/segformer_b2_weights_voc.pth
```

The trained two-class SegFormer-RDA B2 checkpoint is
`logs_improve/best_epoch_weights.pth`. To use a different location:

```powershell
$env:SEGFORMER_RDA_WEIGHTS = 'D:\models\best_epoch_weights.pth'
```

See [weights/README.md](weights/README.md) for the checkpoint details. The
repository uses Git LFS for all `.pth` and `.tif` files.

## Train

From the repository root:

```powershell
python train_improve.py
```

`train.py` remains as a compatibility launcher for the same configuration.
Training outputs are written under `logs_improve/`. To resume
from a checkpoint, set `SEGFORMER_RDA_INIT_WEIGHTS` before starting. To reduce
data-loader processes on a smaller machine, set for example:

```powershell
$env:SEGFORMER_RDA_NUM_WORKERS = '2'
```

The default script automatically uses CUDA when PyTorch reports a CUDA device
and falls back to CPU for basic testing.

## Inference

To run directory prediction on the dataset images:

```powershell
python predict.py
```

The colorized masks are written to `results/prediction_visual/`.

For large GeoTIFF prediction, set the input and output paths and run:

```powershell
$env:SEGFORMER_RDA_BIG_TIFF = 'D:\images\shape.tif'
$env:SEGFORMER_RDA_BIG_TIFF_OUTPUT = 'D:\images\shape_prediction.tif'
python predictbig.py
```

## Evaluation

The evaluation script uses the validation split and writes predictions and
metrics under `results/`:

```powershell
python get_miou.py
```

## Large files and Git LFS

Git LFS must be installed before cloning or pulling the complete data and
weights:

```powershell
git lfs install
git lfs pull
```

Without Git LFS, the repository will contain pointer files instead of the
actual large binaries. GitHub blocks ordinary Git files larger than 100 MiB,
which is why this repository tracks `.pth` and `.tif` with Git LFS.

## Release assets

The recommended release is `v1.0.0`, with optional assets named:

- `segformer-rda-weights-v1.0.0.zip`
- `segformer-rda-dataset-v1.0.0.zip`

The complete files are already tracked by Git LFS in this repository. Release
assets are optional mirrors and are not required for cloning the repository.

## Citation

When using this repository, cite the upstream implementation as well as the
SegFormer-RDA paper or project record associated with the model:

```text
bubbliiiing. segformer-pytorch: SegFormer implementation in PyTorch.
GitHub. https://github.com/bubbliiiing/segformer-pytorch
```
