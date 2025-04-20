import logging
import colorlog


def init_logger(config):
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    if config.simulation_logfile != "None":
        logging.basicConfig(filename=config.simulation_logfile, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    else:
        logging.basicConfig(filename=None, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        fmt_string = '%(log_color)s[%(asctime)s][%(levelname)s]%(message)s'
        # black red green yellow blue purple cyan and white
        log_colors = {
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'purple'
        }
        fmt = colorlog.ColoredFormatter(fmt_string, log_colors=log_colors)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
    return logger
