#!/bin/bash

git clone https://huggingface.co/spaces/52Hz/CMFNet_deblurring

apt-get install wget

wget https://github.com/FanChiMao/CMFNet/releases/download/v0.0/deblur_GoPro_CMFNet.pth -P CMFNet_deblurring/experiments/pretrained_models