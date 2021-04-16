import logging
logger = logging.getLogger(__name__)


def init_logger(name, level=logging.INFO, filename=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-20s   : %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename, mode='w')
        formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-20s   : %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
