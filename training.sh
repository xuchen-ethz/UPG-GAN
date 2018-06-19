#!/bin/bash
python train.py\
    --dataroot ~/data/UPCycGAN/body2image_shirt_and_tshirt/ \
    --model vae_cycle_gan                                   \
    --name shirt_and_tshirt                                 \
