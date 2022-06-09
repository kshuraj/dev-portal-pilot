#!/bin/bash

git clone https://huggingface.co/spaces/akhaliq/JoJoGAN 

cp util.py JoJoGAN/util.py

cd JoJoGAN

pip install -r requirements.txt

mkdir models

gdown https://drive.google.com/uc?id=1jtCg8HQ6RlTmLdnbT2PfW1FJ2AYkWqsK

cp e4e_ffhq_encode.pt models/e4e_ffhq_encode.pt

apt-get install wget

wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 

bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2

mv shape_predictor_68_face_landmarks.dat models/dlibshape_predictor_68_face_landmarks.dat

gdown https://drive.google.com/uc?id=1_cTsjqzD_X9DK3t3IZE53huKgnzj_btZ 