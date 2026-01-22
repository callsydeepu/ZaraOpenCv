import cv2
import numpy as np
import sys

from src.logger import logger
from src.exception import CustomException

def composite_person(
        background_image:np.ndarray,
        aligned_person:np.ndarray,
        aligned_mask:np.ndarray,
        x:int,
        y:int
):
    try:
        logger.info("stage 4:compsiting_starting")
        output=background_image.copy()

        #person dimention

        h,w=aligned_person.shape[:2]

        #roi from bg
        roi=output[y:y+h,x:x+w].astype(np.float32)

        # 1️⃣ CONTACT SHADOW (NEW)
        # ==================================================

        # Convert mask to [0,1]
        shadow = aligned_mask.astype(np.float32) / 255.0

        # Blur heavily → soft shadow
        shadow = cv2.GaussianBlur(shadow, (51, 51), 0)

        # Push shadow slightly downward (gravity cue)
        shadow_offset = 10
        shadow = np.roll(shadow, shadow_offset, axis=0)

        # Shadow strength
        shadow_strength = 0.35

        # Expand to 3 channels
        shadow_3c = cv2.merge([shadow, shadow, shadow])
        shadow_3c=cv2.flip(shadow_3c,1)

        # Darken background under person
        roi = roi * (1 - shadow_strength * shadow_3c)

        #prepare mask
        mask=aligned_mask.astype(np.float32)/255.0

        #expand mask to 3channesl
        mask=cv2.merge([mask,mask,mask])

        mask=cv2.flip(mask,1)
        foreground=aligned_person.astype(np.float32)
        foreground=cv2.flip(foreground,1)
        background=roi.astype(np.float32)

        blended=foreground*mask + background * (1-mask)

        #put result back

        output[y:y+h, x:x+w] = blended.astype(np.uint8)
        
        logger.info("stage 4 completed")
        return output

    except Exception as e:
        logger.info("stage 4 failed")
        raise CustomException(sys,e)