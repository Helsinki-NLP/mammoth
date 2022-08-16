# -*- coding: utf-8 -*-
import os
import json
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger()


def init_logger(
    log_file=None,
    log_file_level=logging.NOTSET,
    rotate=False,
    log_level=logging.INFO,
):
    log_format = logging.Formatter("[%(asctime)s %(process)s %(levelname)s] %(message)s")
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

    return logger


def log_lca_values(step, lca_logs, lca_params, opath, dump_logs=False):
    for k, v in lca_params.items():
        lca_sum = v.sum().item()
        lca_mean = v.mean().item()
        lca_logs[k][f'STEP_{step}'] = {'sum': lca_sum, 'mean': lca_mean}

    if dump_logs:
        if os.path.exists(opath):
            os.system(f'cp {opath} {opath}.previous')
        with open(opath, 'w+') as f:
            json.dump(lca_logs, f)
        logger.info(f'dumped LCA logs in {opath}')
