#!/bin/bash
python test_vae.py \
    --dataroot ./datasets/body2image_suit_and_dress/        \
    --phase test        		                            \
    --model vae_cycle_gan                                   \
    --name suit_and_dress                                   \
    --which_epoch 800                                       \
    --how_many 10 		                                    \
    --n_samples 20                                          \