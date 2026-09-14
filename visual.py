import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

def visualize_masks(mask_dir, output_dir):
               
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

                            
                                
                          
    palette = [
        0, 0, 0,                  
        255, 255, 255,                
        0, 255, 0,                
        0, 0, 255,                
    ] + [0] * (256 * 3 - 12)

    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith('.tif')]
    print(f"Starting conversion of {len(mask_files)} label images...")

    for mask_name in tqdm(mask_files):
        mask_path = os.path.join(mask_dir, mask_name)
        
              
        mask = Image.open(mask_path).convert('P')           
        
                    
        mask.putpalette(palette)
        
                 
        mask.save(os.path.join(output_dir, mask_name))

if __name__ == "__main__":
            
    PROJECT_ROOT = Path(__file__).resolve().parent
    MASK_DIR = Path(os.environ.get(
        "SEGFORMER_RDA_MASK_DIR",
        PROJECT_ROOT / "results" / "detection-results",
    ))
    VIS_DIR = Path(os.environ.get(
        "SEGFORMER_RDA_VIS_DIR",
        PROJECT_ROOT / "results" / "prediction_masks_visual",
    ))
    
    visualize_masks(MASK_DIR, VIS_DIR)
    print(f"Visualization completed! Results saved to: {VIS_DIR}")
