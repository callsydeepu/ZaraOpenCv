import cv2
import numpy as np
import sys

import mediapipe as mp
from src.exception import CustomException
from src.logger import logger

def refine_mask(probability_mask:np.ndarray,threshold:float =0.5) -> np.ndarray:
    try:
        logger.info("stage2:started")
        if probability_mask is None:
            raise ValueError("probab is None")
        
        if len(probability_mask.shape)!=2:
            raise ValueError("probab must be 2D")
        
        #thresholding

        binary_mask=(probability_mask > threshold).astype(np.uint8)*255
        logger.info(f"threshold applied at {threshold}")

        #structing element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        #remove all noise
        binary_mask=cv2.morphologyEx(
            binary_mask,cv2.MORPH_OPEN,kernel
        )

        logger.info("Morph oppening applied")

        # 1️⃣ Close small holes
        refined = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # 2️⃣ Recover thin regions (hands, fingers)
        dist = cv2.distanceTransform(255 - refined, cv2.DIST_L2, 5)
        refined[dist < 4] = 255

        # 3️⃣ Smooth edges
        refined = cv2.GaussianBlur(refined, (7, 7), 0)
        refined = (refined > 127).astype(np.uint8) * 255

        return refined
    except Exception as e:
        logger.error("Stage 2 failed: Mask refinement error")
        raise CustomException(e, sys)
