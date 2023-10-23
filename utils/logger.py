import logging


def make_logger(logfile, rank):
    return CustomLogger(logfile, rank)


class CustomLogger:
    def __init__(self, logfile, rank) -> None:
        self.rank = rank
        if rank == 0:
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            fh = logging.FileHandler(logfile)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(levelname)s - %(message)s', datefmt='%d%m%y-%H%M')
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)
        else:
            self.logger = None

    def debug(self, message):
        if self.rank == 0:
            self.logger.debug(message)

    def info(self, message):
        if self.rank == 0:
            self.logger.info(message)

    def warning(self, message):
        if self.rank == 0:
            self.logger.warning(message)

    def error(self, message):
        if self.rank == 0:
            self.logger.error(message)

    def critical(self, message):
        if self.rank == 0:
            self.logger.critical(message)
