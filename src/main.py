import cv2
import sys

from src.loadimage import load_image
from src.stage1_segmentation import run_stage1_segmentation
from src.stage2_mask_refinement import refine_mask
from src.stage3_alignment import align_person
# from src.stage4_composing import composite_person
from src.logger import logger
from src.exception import CustomException


def main():
    try:
        logger.info("Pipeline started")

        # -------- Load images --------
        person_image = load_image("data/image1.jpg")
        background_image = load_image("data/background.jpeg")
        print("done")

        # -------- Stage 1: Person segmentation (YOLOv8) --------
        probabilty_mask=run_stage1_segmentation(person_image)

        cv2.imshow("Stage 1 - Raw Person Mask", probabilty_mask)
        cv2.waitKey(0)

        # -------- Stage 2: Mask refinement --------
        refined_mask = refine_mask(probabilty_mask)

        cv2.imshow("Stage 2 - Refined Mask", refined_mask)
        cv2.waitKey(0)

        #stage 3:alignment

        aligned_person, aligned_mask, x, y = align_person(
                                person_image,
                                refined_mask,
                                background_image
                            )

        cv2.imshow("Aligned Person", aligned_person)
        cv2.waitKey(0)

        cv2.imshow("Aligned Mask", aligned_mask)
        cv2.waitKey(0)


        preview = background_image.copy()

        h, w = aligned_person.shape[:2]
        preview[y:y+h, x:x+w] = aligned_person

        cv2.imshow("Stage 3 Placement Preview", preview)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

        logger.info("Pipeline completed successfully")

    except CustomException as e:
        logger.error("Pipeline failed")
        print(e)
    except Exception as e:
        logger.error("Unexpected error")
        print(e)


if __name__ == "__main__":
    main()
