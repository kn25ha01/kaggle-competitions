#!/bin/bash

#sleep 60s

CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff0_128x128_5fold.yml
#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff0_192x192_5fold.yml
#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff0_256x256_5fold.yml
#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff0_384x384_5fold.yml

#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff1_128x128_5fold.yml
#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff1_192x192_5fold.yml
#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff1_256x256_5fold.yml
#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff1_384x384_5fold.yml

#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff2_128x128_5fold.yml
#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff2_192x192_5fold.yml
#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff2_256x256_5fold.yml
#CUDA_VISIBLE_DEVICES=0,1 python train.py --config=config/eff2_384x384_5fold.yml

#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff3_192x192_5fold.yml
#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff3_256x256_5fold.yml
#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff3_384x384_5fold.yml

#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff4_128x128_5fold.yml
#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff4_192x192_5fold.yml
#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff4_256x256_5fold.yml
#CUDA_VISIBLE_DEVICES=0,1 python valid.py --config=config/eff4_384x384_5fold.yml



