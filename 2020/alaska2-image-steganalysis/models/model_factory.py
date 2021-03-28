import torch
from efficientnet_pytorch import EfficientNet


def get_efficientnet(model_name, num_classes, in_channels, **kwargs):
    #print(f'model_name : {model_name}')
    #print(f'num_classes: {num_classes}')
    #print(f'in_channels: {in_channels}')
    return EfficientNet.from_pretrained(model_name,
                                        num_classes=num_classes,
                                        in_channels=in_channels)


def get_model(config):
    #print('model name:', config.model.name)
    f = globals().get('get_' + config.model.name)
    if config.model.params is None:
        return f()
    else:
        return f(**config.model.params)


