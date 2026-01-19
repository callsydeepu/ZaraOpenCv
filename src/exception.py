import sys
from src.logger import logger

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    return f"Error in [{exc_tb.tb_frame.f_code.co_filename}] line {exc_tb.tb_lineno}: {str(error)}"

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.message = error_message_detail(error_message, error_detail)
        logger.error(self.message)

    def __str__(self):
        return self.message
