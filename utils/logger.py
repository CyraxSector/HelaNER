import logging
import logging.config

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='a')


def warning(text):
    logging.warning(text)

def error(text):
    logging.error(text)

def debug(text):
    logging.debug(text)
def info(text):
    logging.info(text)


if __name__ =="__main__":
    info("hello")
    debug("heyhey")
    warning("hahah")
    error("hehe")

