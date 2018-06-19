#!/bin/bash
python test_vae.py \
    --dataroot ./datasets/body2image_shirt_and_tshirt//        \
    --dataset_mode unaligned                                \
    --phase test        		                            \
    --model vae_cycle_gan                                   \
    --name shirt_and_tshirt                                   \
    --which_epoch latest                                     \
    --how_many 10 		                                    \
    --n_samples 20                                          \