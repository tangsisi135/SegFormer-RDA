# Dataset

The full source dataset is kept at the repository root in `liefeng/`, matching
the original experiment folder. Its VOC-style layout is:

```text
liefeng/
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

The original folder currently contains 54 full-size image/mask pairs. It does
not include `ImageSets/Segmentation/train.txt` or `val.txt`, so it is not the
canonical 1,350-patch training split described by the training configuration.
Validate any supplied split with:

```powershell
python prepare_dataset_splits.py
```

To use another location, set `SEGFORMER_RDA_DATASET_ROOT` to the directory that
contains `VOC2007`.

The image and mask files are tracked with Git LFS because they are binary
GeoTIFF files.
