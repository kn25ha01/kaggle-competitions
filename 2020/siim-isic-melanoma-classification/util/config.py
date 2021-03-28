# BSD 2-Clause License
# 
# Copyright (c) 2019, pudae
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================

import yaml
from easydict import EasyDict as edict


def _get_default_config():
    c = edict()

    # dataset
    c.data = edict()
    c.data.name = None
    c.data.dir = None
    c.data.params = edict()

    # model
    c.model = edict()
    c.model.name = None
    c.model.params = edict()

    # train
    c.train = edict()
    c.train.out_dir = './output/default_out/'
    c.train.batch_size = None
    c.train.num_epochs = None
    c.train.num_folds = None
    c.train.num_grad_acc = None

    # validation
    c.valid = edict()
    c.valid.batch_size = None

    # test
    c.test = edict()
    c.test.batch_size = None

    # transforms
    c.transform = edict()
    c.transform.name = None
    c.transform.num_preprocessor = 0
    c.transform.params = edict()

    c.seed = 2020

    return c


# _get_default_config()にマージ
def _merge_config(src, dst):
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


# コンフィグ読み込み
def load(config_path):
    with open(config_path, 'r') as fid:
        yaml_config = edict(yaml.load(fid))

    config = _get_default_config()
    _merge_config(yaml_config, config)

    return config
    
