import cv2
import numpy as np
import sys
import mediapipe as mp
from pathlib import Path
from src.logger import logger
from src.exception import CustomException

def run_stage1_segmentation(image:np.ndarray)->np.ndarray:
    try:
        logger.info("stage1_started")
        if image is None:
            raise ValueError("Input image is None")
        
        if len(image.shape)!=3 or image.shape[2]!=3:
            raise ValueError("Input image must be is BGR color image")
        
        #convert BGR-->RGBs
        img_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        logger.info("Converted image from BGR to RGB")

        #load mediapipe selfie segmenttation model
        mp_selfie=mp.solutions.selfie_segmentation
        segmenter=mp_selfie.SelfieSegmentation(model_selection=1)

        #run

        result=segmenter.process(img_rgb)

        if result.segmentation_mask is None:
            raise RuntimeError("Segmentation model returned no mask")
        
        probability_mask=result.segmentation_mask

        logger.info(
            f"Segmentation completed | mask_shape={probability_mask.shape} "
            f"mask_dtype={probability_mask.dtype}"
        )   

        return probability_mask
    except Exception as e:
        logger.error("Stage 1 failed: Person segmentation error")
        raise CustomException(e, sys)

        