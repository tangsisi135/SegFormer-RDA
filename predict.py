#----------------------------------------------------#
                         
                                
#----------------------------------------------------#
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from segformer import SegFormer_Segmentation

PROJECT_ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
                                                 
    #-------------------------------------------------------------------------#
    segformer = SegFormer_Segmentation()
    #----------------------------------------------------------------------------------------------------------#
                      
                                                                         
                                                           
                                                                    
                                                                            
                                                          
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    #-------------------------------------------------------------------------#
                                                     
                                                                 
    #
                                             
    #-------------------------------------------------------------------------#
    count           = False
    name_classes    = ["background","BAO"]
    # name_classes    = ["background","cat","dog"]
    #----------------------------------------------------------------------------------------------------------#
                                                           
                                                                                    
                                                               
                                                                                         
                                       
    #
                                                             
                                            
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
                                                                           
                                        
    #   
                                                  
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
                                            
                                         
    #   
                                                            
    #-------------------------------------------------------------------------#
    dataset_root = Path(os.environ.get(
        "SEGFORMER_RDA_DATASET_ROOT",
        str(PROJECT_ROOT / "data" / "liefeng"),
    ))
    dir_origin_path = str(dataset_root / "VOC2007" / "JPEGImages")
    dir_save_path   = str(PROJECT_ROOT / "results" / "prediction_visual")
    #-------------------------------------------------------------------------#
                                           
                                        
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = str(PROJECT_ROOT / "results" / "models.onnx")

    if mode == "predict":
        '''
        
        Notes for predict.py:
        1. This code cannot directly perform batch prediction. If you want to do batch prediction,
           you can use os.listdir() to iterate through a folder and use Image.open to open image files for prediction.
           For the specific process, please refer to get_miou_prediction.py, which implements the iteration.
        2. If you want to save the result, you can use r_image.save("img.jpg") to save it.
        3. If you do not want the original image and the segmentation map to be blended,
           you can set the blend parameter to False.
        4. If you want to obtain the corresponding region based on the mask, you can refer to the part
           in the detect_image function that uses the prediction results to draw the image,
           determine the class of each pixel, and then obtain the corresponding part based on the class.
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = segformer.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to read camera/video input.")

        fps = 0.0
        while(True):
            t1 = time.time()
                   
            ref, frame = capture.read()
            if not ref:
                break
                           
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                      
            frame = Image.fromarray(np.uint8(frame))
                  
            frame = np.array(segformer.detect_image(frame))
                                  
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = segformer.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        if not os.path.isdir(dir_origin_path):
            raise FileNotFoundError(
                f"Input image directory was not found: {dir_origin_path}. "
                "Download the dataset release or set SEGFORMER_RDA_DATASET_ROOT."
            )
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = segformer.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
                
    elif mode == "export_onnx":
        segformer.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
