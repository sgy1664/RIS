import torch
from torch import nn
import numpy as np
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.nn.functional as F
from model import *
from tqdm import tqdm as tqdm
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # get rid of interpolation warning
from util import *
from util import align_face
import os
from e4e_projection import projection, load_e4e, projection_e4e
import argparse
import sys

#Gives an index to each feature we care about
labels2idx = {
    'nose': 0,
    'eyes': 1,
    'mouth':2,
    'hair': 3,
    'background': 4,
    'cheeks': 5,
    'neck': 6,
    'clothes': 7,
}

# Assign to each feature the cluster index from segmentation
labels_map = {
    0: torch.tensor([7]),
    1: torch.tensor([1,6]),
    2: torch.tensor([4]),
    3: torch.tensor([0,3,5,8,10,15,16]),
    4: torch.tensor([11,13,14]),
    5: torch.tensor([9]),
    6: torch.tensor([17]),
    7: torch.tensor([2,12]),
}

idx2labels = dict((v,k) for k,v in labels2idx.items())
n_class = len(labels2idx)

# compute M given a style code.
@torch.no_grad()
def compute_M(generator, clusterer, w, device='cuda'):
    M = []
    truncation = 0.5
    stop_idx = 11 # choose 32x32 layer to do kmeans clustering
    n_clusters = 18 # Number of Kmeans cluster

    # get segmentation
    _, outputs = generator(w, is_cluster=1)
    cluster_layer = outputs[stop_idx][0]
    activation = flatten_act(cluster_layer)
    seg_mask = clusterer.predict(activation)
    b,c,h,w = cluster_layer.size()

    # create masks for each feature
    all_seg_mask = []
    seg_mask = torch.from_numpy(seg_mask).view(b,1,h,w,1).to(device)

    for key in range(n_class):
        # combine masks for all indices for a particular segmentation class
        indices = labels_map[key].view(1,1,1,1,-1)
        key_mask = (seg_mask == indices.to(device)).any(-1) #[b,1,h,w]
        all_seg_mask.append(key_mask)

    all_seg_mask = torch.stack(all_seg_mask, 1)

    # go through each activation layer and compute M
    for layer_idx in range(len(outputs)):
        layer = outputs[layer_idx][1].to(device)
        b,c,h,w = layer.size()
        layer = F.instance_norm(layer)
        layer = layer.pow(2)

        # resize the segmentation masks to current activations' resolution
        layer_seg_mask = F.interpolate(all_seg_mask.flatten(0,1).float(), align_corners=False,
                                     size=(h,w), mode='bilinear').view(b,-1,1,h,w)

        masked_layer = layer.unsqueeze(1) * layer_seg_mask # [b,k,c,h,w]
        masked_layer = (masked_layer.sum([3,4])/ (h*w))#[b,k,c]

        M.append(masked_layer.to(device))

    M = torch.cat(M, -1) #[b, k, c]

    # softmax to assign each channel to a particular segmentation class
    M = F.softmax(M/.1, 1)
    # simple thresholding
    M = (M>.8).float()

    # zero out torgb transfers, from https://arxiv.org/abs/2011.12799
    for i in range(n_class):
        part_M = style2list(M[:, i])
        for j in range(len(part_M)):
            if j in rgb_layer_idx:
                part_M[j].zero_()
        part_M = list2style(part_M)
        M[:, i] = part_M

    return M

def main(args):

    path = os.getcwd()
    input_dir = os.path.join(path,args.input_dir)
    output_dir = os.path.join(path,args.output_dir)
    print(input_dir, output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Load pretrained generator
    device = 'cuda' # if GPU memory is low, use cpu instead
    generator = Generator(1024, 512, 8, channel_multiplier=2).to(device).eval()

    # load model file from current directory
    ensure_checkpoint_exists('models/stylegan2-ffhq-config-f.pt')
    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"], strict=False)
    with torch.no_grad():
        mean_latent = generator.mean_latent(50000)

    # load pre generated catalog
    clusterer = pickle.load(open("models/catalog.pkl", "rb"))

    im_path1 = os.path.join(args.input_dir, args.im_path1)
    im_path2 = os.path.join(args.input_dir, args.im_path2)
    im_name1 = os.path.splitext(args.im_path1)[0]
    im_name2 = os.path.splitext(args.im_path2)[0]

    # aligns and crops face
    cropped_face1 = align_face(im_path1)
    cropped_face2 = align_face(im_path2)

    net_e4e = load_e4e(device=device)
    projection_e4e(net_e4e, cropped_face1, im_name1, generator, output_dir, device)
    projection_e4e(net_e4e, cropped_face2, im_name2, generator, output_dir, device)

    #projection(cropped_face1, im_name1, generator, output_dir, device)
    #projection(cropped_face2, im_name2, generator, output_dir, device)

    source = load_source([im_name1], generator, output_dir, device)
    #source = load_source(['brad_pitt'], generator, device)
    source_im, _ = generator(source)
    display_image(source_im, size=256)
    
    ref = load_source([im_name2], generator,output_dir, device)
    #ref = load_source(['emma_watson', 'emma_stone', 'jennie'], generator, output_dir, device)
    ref_im, _ = generator(ref)
    ref_im = downsample(ref_im)

    if args.display:
        save_name = os.path.join(output_dir,im_name2+"_References.jpg")
        show(normalize_im(ref_im).permute(0,2,3,1).cpu(), title='References',save=True, save_path=save_name)
    # Compute M for both source and reference images use cpu here to save memory
    source_M = compute_M(generator,clusterer, source, device='cpu')
    ref_M = compute_M(generator,clusterer, ref, device='cpu')

    # Find relevant channels for source and reference by taking max over their individual M
    max_M = torch.max(source_M.expand_as(ref_M), ref_M)
    max_M = add_pose(max_M, labels2idx)

    all_im = {}

    with torch.no_grad():
        # features we are interest in transferring
        parts=[]
        if args.styletype == 'all':
            parts = ['eyes', 'nose', 'mouth', 'hair','pose']
        else:
            parts = [args.styletype]

        for label in parts:
            if label == 'pose':
                idx = -1
            else:
                idx = labels2idx[label]

            part_M = max_M[:,idx].to(device)
            blend = style2list(add_direction(source, ref, part_M, 1.3))

            blend_im, _ = generator(blend)
            blend_im = downsample(blend_im).cpu()
            all_im[label] = normalize_im(blend_im)
            if args.save:
                save_name = os.path.join(output_dir,im_name1+"_"+im_name2+"_"+label+".jpg")
                show(normalize_im(blend_im).permute(0,2,3,1).cpu(), title='References',save=True, save_path=save_name)
                #save_image(blend_im,title=save_name)
    if args.display:
        save_name = os.path.join(output_dir,im_name1+"_"+im_name2+".jpg")
        part_grid(normalize_im(source_im.detach()), normalize_im(ref_im.detach()), all_im,save=True,save_path=save_name);

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RIS')

    # I/O arguments
    parser.add_argument('--input_dir', type=str, default='input/face',
                        help='The directory of the images to be inverted')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='The directory to save the latent codes and inversion images')
    parser.add_argument('--im_path1', type=str, default='00018.jpg', help='Identity image')
    parser.add_argument('--im_path2', type=str, default='15.png', help='Reference image')
    parser.add_argument('--display', type=bool, default=True, help='show the results')
    parser.add_argument('--save', type=bool, default=True, help='save the results')
    parser.add_argument('--styletype', type=str, default='hair', help='style type transfer',choices=['eyes','nose','mouth','hair','pose','all'])

    # StyleGAN2 setting
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default="models/stylegan2-ffhq-config-f.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)

    args = parser.parse_args()
    print(args)
    sys.exit(main(args))
