import logging, sys

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger: 
    logger = logging.getLogger(name) 
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(ch)
    return logger
