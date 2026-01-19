import cv2
from pathlib import Path


from src.logger import logger
from src.exception import CustomException

def load_image(path:str):

    try:
        image_path=Path(path)
        if not image_path.exists():
            raise CustomException("Path doesnt exists!!")
        
        logger.info("Reading image")

        image=cv2.imread(str(image_path))
        
        if image is None:
            raise CustomException("failed to load imge")
        
        logger.info("image loaded succesfully")

        return image
    except Exception as e:
        logger.error("image loading failed")
        raise CustomException("Image loading failed") from e





