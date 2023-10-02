# -*- coding: utf-8 -*-
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Union

logger = logging.getLogger()


def init_logger(
    log_file=None,
    log_file_level=logging.NOTSET,
    rotate=False,
    log_level=logging.INFO,
    gpu_id='',
    structured_log_file=None,
):
    log_format = logging.Formatter(f"[%(asctime)s %(process)s {gpu_id} %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        if rotate:
            file_handler = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=10)
        else:
            file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    if structured_log_file:
        init_structured_logger(structured_log_file)

    return logger


def init_structured_logger(
    log_file=None,
):
    # Log should be parseable as a jsonl file. Format should not include anything extra.
    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger("structured_logger")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='a', delay=True)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.handlers = [file_handler]
    logger.propagate = False


def structured_logging(obj: Dict[str, Union[str, int, float]]):
    structured_logger = logging.getLogger("structured_logger")
    if not structured_logger.hasHandlers:
        return
    try:
        structured_logger.info(json.dumps(obj))
    except Exception:
        pass
