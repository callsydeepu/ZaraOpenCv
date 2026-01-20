import cv2
import numpy as np
import sys

from src.exception import CustomException
from src.logger import logger

def align_person(
        person_image:np.ndarray,
        person_masked:np.ndarray,
        bg_image:np.ndarray,
        scale_ratio:float=1
):
    try:
        logger.info("stage3-alignmen Started")
        coords=cv2.findNonZero(person_masked)
        if coords is None:
            raise ValueError("No foreground pixels found in mask")
        
        x,y,w,h=cv2.boundingRect(coords)

        logger.info("bounded successfully")

        #crop person and mask
        crop_person=person_image[y:y+h,x:x+w]
        crop_mask=person_masked[y:y+h,x:x+w]

        logger.info("croped person and croped masked person")

        #compute scale relative to background
        bg_h,bg_w=bg_image.shape[:2]

        logger.info(f"bg height and width is {bg_h,bg_w}")

        target_person_height=int(bg_h*scale_ratio)
        scale=target_person_height/h
        new_w=int(w*scale)
        new_h=int(h*scale)

        #resize person and mask

        aligned_person=cv2.resize(
            crop_person,(new_w,new_h),interpolation=cv2.INTER_LINEAR
        )

        aligned_mask=cv2.resize(
            crop_mask,(new_w,new_h),interpolation=cv2.INTER_NEAREST
        )

        logger.info("Person resized relative to background")

        #placement

        top_left_x=(bg_w-new_w)//2
        top_left_y=bg_h-new_h

        logger.info(
            f"Alignment complete | position=({top_left_x}, {top_left_y})"
        )
        return aligned_person, aligned_mask, top_left_x, top_left_y


    except Exception as e:
        logger.error("Stage 3 failed: Alignment error")
        raise CustomException(e, sys)
    