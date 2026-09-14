# Dataset

The dataset is intentionally not included in the source-code repository. Put
the dataset downloaded from the GitHub Release at `data/liefeng/` with this
VOC-style layout:

```text
data/liefeng/
└── VOC2007/
    ├── JPEGImages/
    │   └── <patch-id>.tif
    ├── SegmentationClass/
    │   └── <patch-id>.tif
    └── ImageSets/
        └── Segmentation/
            ├── train.txt
            └── val.txt
```

The canonical split contains 1,147 training patches and 203 validation
patches. Validate a downloaded dataset with:

```powershell
python prepare_dataset_splits.py
```

To use another location, set `SEGFORMER_RDA_DATASET_ROOT` to the directory that
contains `VOC2007`.

Important: the source folder inspected while preparing this release currently
contains 54 full-size image/mask pairs and no `ImageSets/Segmentation/train.txt`
or `val.txt`. Those files are not a substitute for the canonical 1,350-patch
dataset. Do not publish that incomplete source dataset as the training Release
asset until the patch extraction and split files have been supplied and
validated.
