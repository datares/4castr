import os
import logging
import colorlog

def init_logger(dunder_name, show_debug=False) -> logging.Logger:
    log_format = (
        '%(asctime)s - '
        '%(name)s - '
        '%(funcName)s - '
        '%(levelname)s - '
        '%(message)s'
    )
   # bold_seq = '\033[1m'
   # colorlog_format = (
   #     f'{bold_seq} '
   #     '%(log_color)s '
   #     f'{log_format}'
   # )
   # colorlog.basicConfig(format=colorlog_format)
    logging.getLogger('tensorflow').disabled = True
    logger = logging.getLogger(dunder_name)

    if show_debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    return logger
