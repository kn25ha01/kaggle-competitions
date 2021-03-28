#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py --config=configs/eff0_224_RBG_JMiPOD.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config=configs/eff0_224_RBG_JUNIWARD.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config=configs/eff0_224_RBG_UERD.yml

CUDA_VISIBLE_DEVICES=0 python train.py --config=configs/eff0_224_DCT_JMiPOD.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config=configs/eff0_224_DCT_JUNIWARD.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config=configs/eff0_224_DCT_UERD.yml

