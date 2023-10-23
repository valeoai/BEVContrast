import yaml
from easydict import EasyDict


def log_config(config, logger):
    for key, val in config.items():
        if isinstance(val, EasyDict):
            logger.info("===== %s =====:" % key)
            log_config(val, logger)
        else:
            logger.info('%s: %s' % (key, val))


def generate_config(config):
    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    return EasyDict(config)
