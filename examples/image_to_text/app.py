"""
Hello World:
A small program to demonstrate how the app package is supposed to be organised.
Here we've used 'bert-base-uncased and bert-base-cased' models from HuggingFace to caption the image of user's
input.
"""
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_model(image):
    """
    Example taken from: https://huggingface.co/spaces/Salesforce/BLIP
    """
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
    
    image = Image.open(image)
    image = image.convert('RGB')
    timage = transform(image).unsqueeze(0).to(device) 
    with torch.no_grad():
        caption = model.generate(timage, sample=False, num_beams=3, max_length=20, min_length=5)
        # print(caption[0])
    return caption[0]

def main(input_image):
    """
    Argument(s) name in the function signature are matching with the id in the
    app.config
    """
    output = run_model(input_image)
    print(output)
    return output

# if __name__=="__main__":
#     main('download.jpg')
