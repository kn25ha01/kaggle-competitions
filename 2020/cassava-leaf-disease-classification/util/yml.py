import yaml

def get_config(yml_path, logger=None):
    if logger:
        logger.info(f'open {yml_path}')
    with open(yml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
