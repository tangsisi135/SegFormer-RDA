import os
from pathlib import Path
import numpy as np
import math
import torch
from nets.segformer_rda import SegFormer
import gc

          
def get_model(model_path, num_classes=2, phi="b2", device="cpu"):
    """ Initialize the model and load the weights only once """
    model = SegFormer(num_classes=num_classes, phi=phi)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_and_write():
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise ImportError(
            "predictbig.py requires GDAL. Install it separately, for example "
            "with conda-forge: conda install -c conda-forge gdal."
        ) from exc

    gdal.UseExceptions()
    project_root = Path(__file__).resolve().parent
    TifPath = os.environ.get(
        "SEGFORMER_RDA_BIG_TIFF",
        str(project_root / "input" / "shape.tif"),
    )
    model_path = os.environ.get(
        "SEGFORMER_RDA_WEIGHTS",
        str(project_root / "weights" / "best_epoch_weights.pth"),
    )
    ResultPath = os.environ.get(
        "SEGFORMER_RDA_BIG_TIFF_OUTPUT",
        str(project_root / "results" / "shape_prediction.tif"),
    )
    
              
    Path(ResultPath).parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(3, 1, 1)
    
               
    if not os.path.isfile(TifPath):
        raise FileNotFoundError(
            f"Input GeoTIFF was not found: {TifPath}. Set SEGFORMER_RDA_BIG_TIFF."
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model weights were not found: {model_path}. Set SEGFORMER_RDA_WEIGHTS."
        )

    ds = gdal.Open(TifPath)
    width = ds.RasterXSize
    height = ds.RasterYSize
    geotrans = ds.GetGeoTransform()
    proj = ds.GetProjection()
    
                   
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(ResultPath, width, height, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(geotrans)
    out_ds.SetProjection(proj)
    out_band = out_ds.GetRasterBand(1)

                                     
    model = get_model(model_path, num_classes=2, phi="b2", device=device)

                      
    SideLength = 512
    RepetitiveLength = int((1 - math.sqrt(0.6)) * SideLength / 2)
    Step = SideLength - RepetitiveLength * 2
    
    ColumnNum = int((height - RepetitiveLength * 2) / Step)
    RowNum = int((width - RepetitiveLength * 2) / Step)
    
    ColumnOver = (height - RepetitiveLength * 2) % Step + RepetitiveLength
    RowOver = (width - RepetitiveLength * 2) % Step + RepetitiveLength

    print(f"Starting prediction: image size {width}x{height}, processing in tiles...")

                      
    with torch.inference_mode(): 
        for i in range(ColumnNum + 1):
            for j in range(RowNum + 1):
                                     
                if i < ColumnNum:
                    y_off = i * Step
                else:
                    y_off = height - SideLength
                
                if j < RowNum:
                    x_off = j * Step
                else:
                    x_off = width - SideLength

                                   
                tile = ds.ReadAsArray(x_off, y_off, SideLength, SideLength)
                
                                                            
                if len(tile.shape) == 3:
                    tile = tile.transpose(1, 2, 0)
                
                                                   
                if tile.ndim == 2:
                    tile = np.repeat(tile[..., None], 3, axis=2)
                tile = tile[..., :3].astype(np.float32)
                img_tensor = torch.from_numpy(tile.transpose(2, 0, 1)).float()
                img_tensor = ((img_tensor - mean) / std).unsqueeze(0).to(device)
                out = model(img_tensor)
                pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                
                                            
                                     
                final_pred = (pred == 1).astype(np.uint8) * 255
                                                             
                
                                          
                target_x, target_y = x_off + RepetitiveLength, y_off + RepetitiveLength
                source_x, source_y = RepetitiveLength, RepetitiveLength
                write_w, write_h = SideLength - 2 * RepetitiveLength, SideLength - 2 * RepetitiveLength

                if j == 0:
                    target_x, source_x = x_off, 0
                    write_w = SideLength - RepetitiveLength
                elif j == RowNum:
                    target_x, source_x = width - RowOver, SideLength - RowOver
                    write_w = RowOver
                
                if i == 0:
                    target_y, source_y = y_off, 0
                    write_h = SideLength - RepetitiveLength
                elif i == ColumnNum:
                    target_y, source_y = height - ColumnOver, SideLength - ColumnOver
                    write_h = ColumnOver

                      
                tile_to_write = final_pred[source_y : source_y + write_h, source_x : source_x + write_w]
                out_band.WriteArray(tile_to_write, target_x, target_y)

                  
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            if (i + 1) % 10 == 0:
                print(f"Progress: Row {i+1}/{ColumnNum+1} processed")

             
    out_band.FlushCache()
    del ds, out_ds
    print("Prediction and writing completed! Results saved to:", ResultPath)

if __name__ == "__main__":
    predict_and_write()
