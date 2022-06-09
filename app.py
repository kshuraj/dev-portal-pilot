from transformers import pipeline
import sys
sys.path.append("JoJoGAN")
import os
from datetime import datetime
from typing import overload
import pathlib
from io import BytesIO
from PIL import Image
import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from JoJoGAN.util import *
from PIL import Image
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
from JoJoGAN.model import *
from copy import deepcopy
import imageio
import os
import sys
import numpy as np
import torchvision.transforms as transforms
from argparse import Namespace
from JoJoGAN.e4e.models.psp import pSp
from huggingface_hub import hf_hub_download


device= 'cpu'
model_path = 'JoJoGAN/models/e4e_ffhq_encode.pt'
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
opts= Namespace(**opts)
net = pSp(opts, device).eval().to(device)


@ torch.no_grad()
def projection(img, name, device='cpu'):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    img = transform(img).unsqueeze(0).to(device)
    images, w_plus = net(img, randomize_noise=False, return_latents=True)
    result_file = {}
    result_file['latent'] = w_plus[0]
    torch.save(result_file, name)
    return w_plus[0]

def run_model(image,model):
    latent_dim = 512

    original_generator = Generator(1024, latent_dim, 8, 2).to(device)
    ckpt = torch.load('JoJoGAN/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    original_generator.load_state_dict(ckpt["g_ema"], strict=False)
    mean_latent = original_generator.mean_latent(10000)

    generatorjojo = deepcopy(original_generator)

    generatordisney = deepcopy(original_generator)

    generatorjinx = deepcopy(original_generator)

    generatorcaitlyn = deepcopy(original_generator)

    generatoryasuho = deepcopy(original_generator)

    generatorarcanemulti = deepcopy(original_generator)

    generatorart = deepcopy(original_generator)

    generatorspider = deepcopy(original_generator)


    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


    modeljojo = hf_hub_download(repo_id="akhaliq/JoJoGAN-jojo", filename="jojo_preserve_color.pt")

    ckptjojo = torch.load(modeljojo, map_location=lambda storage, loc: storage)
    generatorjojo.load_state_dict(ckptjojo["g"], strict=False)

    modeldisney = hf_hub_download(repo_id="akhaliq/jojogan-disney", filename="disney_preserve_color.pt")

    ckptdisney = torch.load(modeldisney, map_location=lambda storage, loc: storage)
    generatordisney.load_state_dict(ckptdisney["g"], strict=False)

    modeljinx = hf_hub_download(repo_id="akhaliq/jojo-gan-jinx", filename="arcane_jinx_preserve_color.pt")

    ckptjinx = torch.load(modeljinx, map_location=lambda storage, loc: storage)
    generatorjinx.load_state_dict(ckptjinx["g"], strict=False)

    modelcaitlyn = hf_hub_download(repo_id="akhaliq/jojogan-arcane", filename="arcane_caitlyn_preserve_color.pt")

    ckptcaitlyn = torch.load(modelcaitlyn, map_location=lambda storage, loc: storage)
    generatorcaitlyn.load_state_dict(ckptcaitlyn["g"], strict=False)

    modelyasuho = hf_hub_download(repo_id="akhaliq/JoJoGAN-jojo", filename="jojo_yasuho_preserve_color.pt")

    ckptyasuho = torch.load(modelyasuho, map_location=lambda storage, loc: storage)
    generatoryasuho.load_state_dict(ckptyasuho["g"], strict=False)

    model_arcane_multi = hf_hub_download(repo_id="akhaliq/jojogan-arcane", filename="arcane_multi_preserve_color.pt")

    ckptarcanemulti = torch.load(model_arcane_multi, map_location=lambda storage, loc: storage)
    generatorarcanemulti.load_state_dict(ckptarcanemulti["g"], strict=False)

    modelart = hf_hub_download(repo_id="akhaliq/jojo-gan-art", filename="art.pt")

    ckptart = torch.load(modelart, map_location=lambda storage, loc: storage)
    generatorart.load_state_dict(ckptart["g"], strict=False)

    modelSpiderverse = hf_hub_download(repo_id="akhaliq/jojo-gan-spiderverse", filename="Spiderverse-face-500iters-8face.pt")

    ckptspider = torch.load(modelSpiderverse, map_location=lambda storage, loc: storage)
    generatorspider.load_state_dict(ckptspider["g"], strict=False)

    output_path = os.path.join('static/images')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 

    # image = Image.open(BytesIO(image))
    image = Image.open(image)
    image = image.convert('RGB')
    temp_file = os.path.join(output_path,'out.jpg')
    image.save(temp_file)

    aligned_face = align_face(temp_file)

    my_w = projection(aligned_face, "test.pt", device).unsqueeze(0)
    if model == 'JoJo':
        with torch.no_grad():
            my_sample = generatorjojo(my_w, input_is_latent=True)  
    elif model == 'Disney':
        with torch.no_grad():
            my_sample = generatordisney(my_w, input_is_latent=True)
    elif model == 'Jinx':
        with torch.no_grad():
            my_sample = generatorjinx(my_w, input_is_latent=True)
    elif model == 'Caitlyn':
        with torch.no_grad():
            my_sample = generatorcaitlyn(my_w, input_is_latent=True)
    elif model == 'Yasuho':
        with torch.no_grad():
            my_sample = generatoryasuho(my_w, input_is_latent=True)
    elif model == 'Arcane Multi':
        with torch.no_grad():
            my_sample = generatorarcanemulti(my_w, input_is_latent=True)
    elif model == 'Art':
        with torch.no_grad():
            my_sample = generatorart(my_w, input_is_latent=True)
    else:
        with torch.no_grad():
            my_sample = generatorspider(my_w, input_is_latent=True)
            
    
    npimage = my_sample[0].permute(1, 2, 0).detach().numpy()

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    output_file = os.path.join(output_path, 'output.jpeg')
    imageio.imwrite(output_file, npimage)
    return output_file


def main(input_image,model):
    """
    Argument(s) name in the function signature are matching with the id in the
    app.config
    """
    output = run_model(input_image,model)
    return output


# if __name__=="__main__":
#     main('mona.png',"JoJo")