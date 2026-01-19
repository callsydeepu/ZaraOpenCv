import cv2
import sys
import numpy as np

from src.loadimage import load_image
from src.stage1_segmentation import run_stage1_segmentation
from src.logger import logger
from src.exception import CustomException


def main():
    try:
        logger.info("Pipeline started")

        # Step 1: Load image
        image = load_image("data/person.jpeg")

        # Step 2: Run Stage 1
        probability_mask = run_stage1_segmentation(image)

        # Step 3: Visualize the mask
        mask_visual = (probability_mask * 255).astype("uint8")

        cv2.imshow("Original Image", image)
        cv2.imshow("Stage 1 - Person Probability Mask", mask_visual)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        logger.info("Stage 1 test completed successfully")

    except CustomException as e:
        logger.error("Pipeline failed")
        print(e)


if __name__ == "__main__":
    main()
