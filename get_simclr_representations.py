'''
Based on https://github.com/ae-foster/pytorch-simclr/blob/simclr-master/gradient_linear_clf.py
'''


import os
import numpy as np
import argparse
from collections import Counter
import pdb

import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from resnet import get_resnet, name_to_params


class ImagenetValidationDataset(Dataset):
    def __init__(self, val_path, ground_truth_path): # Modified to take in separate path for ground truth
        super().__init__()
        self.val_path = val_path
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        with open(ground_truth_path) as f:
            self.labels = [int(l) - 1 for l in f.readlines()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.val_path, f'ILSVRC2012_val_{item + 1:08d}.JPEG')).convert('RGB')
        return self.transform(img), self.labels[item]

@torch.no_grad()
def run(pth_path):
    device = 'cuda'
    dataset = ImagenetValidationDataset('/work/data/imagenet/val/val', '/home/eecs/tiffany_ding/data/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt') # MODIFIED
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=0)
    net, _ = get_resnet(*name_to_params(pth_path)) # renamed model --> net

    print('==> loading encoder from checkpoint..')
    net.load_state_dict(torch.load(pth_path)['resnet'])
    net = net.to(device)

    #net = torch.nn.dataparallel(net, device_ids = [0,1])
    #torch.backends.cudnn.benchmark = true
    # if device == 'cuda':
    #     repr_dim = net.channels_out
    #     net = torch.nn.dataparallel(net)
    #     net.representation_dim = repr_dim
    #     torch.backends.cudnn.benchmark = true

    net.eval()

    features = [] 
    labels = []
        
    t = tqdm(enumerate(dataloader), total=len(dataloader), bar_format='{desc}{bar}{r_bar}')
    for batch_idx, (inputs, targets) in t:
        inputs = inputs.to(device)
        representation = net(inputs)
        features += [representation.cpu()]
        labels += [targets.cpu()]

    features = torch.cat(features,dim=0)
    labels = torch.cat(labels,dim=0)
    
    save_to = '/home/eecs/tiffany_ding/code/simclrv2-pytorch/.cache/simclr_representations/imagenet_val'
    torch.save(features,save_to + '_features.pt')
    torch.save(labels,save_to + '_labels.pt')
    print('Saved features and labels to', save_to)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimCLR verifier')
    parser.add_argument('--pth_path', type=str, default='r152_3x_sk1.pth',  help='path of the input checkpoint file')
    args = parser.parse_args()
    run(args.pth_path)


#####


