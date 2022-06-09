"""
Hello World:
A small program to demonstrate how the app package is supposed to be organised.
Here we've used 'gpt2' model from HuggingFace to generate text for user's
input.
"""
from transformers import pipeline
import sys
sys.path.append("BLIP")
import os
from datetime import datetime
from io import BytesIO
from typing import List, Optional
from PIL import Image
import pathlib
import imageio
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
from models.blip_vqa import blip_vqa

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_model(image,model_sel,question):
    '''
        Input : Image,Model_sel,Question
        Output : Text
        Function : This function accepts image, model and question to describe the image or answer to the question by the user.
                    The model selection has two types "Image Captioning" and "Visual Question Answering".
                    "Image Captioning" will be the default value.
                    "quetion" parameter is optional here since only when Visual Question Answering is choosen this user input is considered.
    '''
    image_size = 384
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
        
    model = blip_decoder(pretrained=model_url, image_size=384, vit='base',med_config='BLIP/configs/med_config.json')
    model.eval()
    model = model.to(device)

    model_url_vq = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_vqa.pth'
            
    model_vq = blip_vqa(pretrained=model_url_vq, image_size=480, vit='base')
    model_vq.eval()
    model_vq = model_vq.to(device)
    
    image = Image.open(image)
    image = image.convert('RGB')

    if model_sel == 'Image Captioning':
        timage = transform(image).unsqueeze(0).to(device) 
        with torch.no_grad():
            caption = model.generate(timage, sample=False, num_beams=3, max_length=20, min_length=5)
        return 'caption: '+caption[0]

    else:
        image_vq = transform(image).unsqueeze(0).to(device) 
        with torch.no_grad():
            answer = model_vq(image_vq, question, train=False, inference='generate') 
        return  'answer: '+answer[0]

def main(input_image,model_sel,question):
    """
    Argument(s) name in the function signature are matching with the id in the
    app.config
    """
    output = run_model(input_image,model_sel,question)
    return output