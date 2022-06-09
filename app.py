"""
Hello World:
A small program to demonstrate how the app package is supposed to be organised.
Here we've used 'gpt2' model from HuggingFace to generate text for user's
input.
"""
from transformers import pipeline
import sys
sys.path.append("CMFNet_deblurring")
import os
from datetime import datetime
from typing import overload
import pathlib
from io import BytesIO
from PIL import Image
import imageio

def run_model(image):
    """
    Example taken from: https://huggingface.co/gpt2#how-to-use
    """
    input_path = os.path.join('static/images/input')
    output_path = os.path.join('static/images')
    pathlib.Path(input_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    basewidth = 512
    img = Image.open(image)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.BILINEAR)
    temp_file = os.path.join(input_path, 'out.png')
    img.save(temp_file)

    os.system(f'python CMFNet_deblurring/main_test_CMFNet.py --input_dir {input_path} --result_dir {output_path} --weights CMFNet_deblurring/experiments/pretrained_models/deblur_GoPro_CMFNet.pth')

    output_file = os.path.join(output_path, 'out.png')
    return output_file

def main(input_image):
    """
    Argument(s) name in the function signature are matching with the id in the
    app.config
    """
    output = run_model(input_image)
    # print(output)
    return output


# if __name__=="__main__":
#     main('download.png')