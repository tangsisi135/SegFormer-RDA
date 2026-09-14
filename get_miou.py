import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from segformer import SegFormer_Segmentation
from utils.utils_metrics import compute_mIoU, show_results

PROJECT_ROOT = Path(__file__).resolve().parent

'''

'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
                                
                                                
                              
                              
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
                   
    #------------------------------#
    num_classes     = 2
    #--------------------------------------------#
                                   
    #--------------------------------------------#
    #name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    name_classes    = ["_background_","feng"]
    #-------------------------------------------------------#
                      
                       
    #-------------------------------------------------------#
    VOCdevkit_path  = os.environ.get(
        "SEGFORMER_RDA_DATASET_ROOT",
        str(PROJECT_ROOT / "data" / "liefeng"),
    )

    # The project uses the original patch-level train/validation split.
    # Evaluate the held-out validation patches because no test split is kept.
    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines()
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = str(PROJECT_ROOT / 'results')
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        segformer = SegFormer_Segmentation()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".tif")
            image       = Image.open(image_path)
            image       = segformer.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".tif"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)               
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
