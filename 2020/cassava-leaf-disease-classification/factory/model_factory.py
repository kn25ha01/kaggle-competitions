from efficientnet_pytorch import EfficientNet

#TODO:add Vision Transformer

def get_efficientnet(config: dict, logger=None):
    model_name = config['model_name']
    in_channels = config['in_channels']
    num_classes = config['num_classes']

    model = EfficientNet.from_pretrained(
        model_name=model_name,
        weights_path=None,
        advprop=False,
        in_channels=in_channels,
        num_classes=num_classes,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if logger:
        logger.debug(f"model name       : {model_name}")
        logger.debug(f"input channels   : {in_channels}")
        logger.debug(f"output classes   : {num_classes}")
        logger.debug(f"trainable params : {trainable_params}")

    return model


def get_model(config: dict, logger=None):
    f = globals().get('get_' + config['name'])
    return f(config['params'], logger)


if __name__ == "__main__":
    import yaml

    from logging import getLogger, basicConfig
    logger = getLogger(__name__)
    basicConfig(level='DEBUG')

    yml_path1 = './yml/model01_1/version00.yml'
    with open(yml_path1, 'r') as f:
        config1 = yaml.load(f, Loader=yaml.FullLoader)
    model_config1 = config1['model']
    model1 = get_model(model_config1, logger)

    yml_path2 = './yml/model02_1/version00.yml'
    with open(yml_path2, 'r') as f:
        config2 = yaml.load(f, Loader=yaml.FullLoader)
    model_config2 = config2['model']
    model2 = get_model(model_config2, logger)

    yml_path3 = './yml/model02_2/version00.yml'
    with open(yml_path3, 'r') as f:
        config3 = yaml.load(f, Loader=yaml.FullLoader)
    model_config3 = config3['model']
    model3 = get_model(model_config3, logger)
