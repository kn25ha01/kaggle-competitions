import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class Model(nn.Module):
    def __init__(self, arch, num_metas: int):
        super(Model, self).__init__()
        self.arch = arch
        #if 'EfficientNet' in str(arch.__class__):
        #    self.arch._fc = nn.Linear(in_features=1280, out_features=500, bias=True)
        self.meta = nn.Sequential(nn.Linear(num_metas, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(500, 250),  # FC layer output will have 250 features
                                  nn.BatchNorm1d(250),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        self.ouput = nn.Linear(self.arch._fc.out_features + 250, 1)
        
    def forward(self, inputs):
        img, meta = inputs
        cnn_features = self.arch(img)
        meta_features = self.meta(meta)
        features = torch.cat((cnn_features, meta_features), dim=1)
        output = self.ouput(features)
        return output


def get_efficientnet(model_name, num_classes, in_channels, num_metas, **kwargs):
    #print(f'model_name : {model_name}')
    #print(f'num_classes: {num_classes}')
    #print(f'in_channels: {in_channels}')
    arch = EfficientNet.from_pretrained(model_name)
    model = Model(arch=arch, num_metas=num_metas)
    return model


def get_model(config):
    #print(f'config.model.name: {config.model.name}')
    f = globals().get('get_' + config.model.name)
    if config.model.params is None:
        return f()
    else:
        return f(**config.model.params)


