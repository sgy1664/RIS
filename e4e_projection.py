import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from argparse import Namespace
from e4e.models.psp import pSp
from util import *


@ torch.no_grad()
def projection(img, name, generator, output='inversion_codes', device='cuda', model_path='models/e4e_ffhq_encode.pt'):
    ensure_checkpoint_exists(model_path)
    print("loading model",model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts, device).eval().to(device)

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
    filename = output+"/" + name + '.pt'
    result_file['latent'] = w_plus[0]
    torch.save(result_file, filename)

@ torch.no_grad()
def load_e4e(model_path='models/e4e_ffhq_encode.pt', device='cuda'):
    ensure_checkpoint_exists(model_path)
    print("loading model",model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts, device).eval().to(device)
    return net

@ torch.no_grad()
def projection_e4e(net, img, name, generator, output='inversion_codes',device='cuda'):
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
    filename = output+"/" + name + '.pt'
    result_file['latent'] = w_plus[0]
    torch.save(result_file, filename)

