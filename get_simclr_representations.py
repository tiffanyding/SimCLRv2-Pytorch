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


# def accuracy(output, target, topk=(1,)):
#     maxk = max(topk)
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t().cpu()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum().item()
#         res.append(correct_k)
#     return res

# Copied from https://github.com/ae-foster/pytorch-simclr/blob/simclr-master/evaluate/lbfgs.py
def test(testloader, device, net, clf):
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    clf.eval()
    test_clf_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        t = tqdm(enumerate(testloader), total=len(testloader), desc='Loss: **** | Test Acc: ****% ',
                 bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (inputs, targets) in t:
            inputs, targets = inputs.to(device), targets.to(device)
            representation = net(inputs)
            # test_repr_loss = criterion(representation, targets)
            raw_scores = clf(representation)
            clf_loss = criterion(raw_scores, targets)

            test_clf_loss += clf_loss.item()
            _, predicted = raw_scores.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            t.set_description('Loss: %.3f | Test Acc: %.3f%% ' % (test_clf_loss / (batch_idx + 1), 100. * correct / total))

    acc = 100. * correct / total
    return acc

@torch.no_grad()
def run(pth_path):
    device = 'cuda'
    dataset = ImagenetValidationDataset('/work/data/imagenet/val/val', '/home/eecs/tiffany_ding/data/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt') # MODIFIED
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=0)
    # testloader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8) # TODO: Figure out what dataset I should be testing on (currently testing on the train set...)
    net, _ = get_resnet(*name_to_params(pth_path)) # renamed model --> net

    print('==> Loading encoder from checkpoint..')
    net.load_state_dict(torch.load(pth_path)['resnet'])
    net = net.to(device)

    #net = torch.nn.DataParallel(net, device_ids = [0,1])
    #torch.backends.cudnn.benchmark = True
    # if device == 'cuda':
    #     repr_dim = net.channels_out
    #     net = torch.nn.DataParallel(net)
    #     net.representation_dim = repr_dim
    #     torch.backends.cudnn.benchmark = True

    # Define linear classifier and loss function
    # num_classes = 1000
    # clf = torch.nn.Linear(net.representation_dim, num_classes).to(device)
    
    # criterion = torch.nn.CrossEntropyLoss()
    # clf_optimizer = torch.optim.SGD(clf.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov,
    #                       weight_decay=args.weight_decay)


    net.eval()

    # placeholder for batch features
    features = [] 
        
    t = tqdm(enumerate(dataloader), total=len(dataloader), bar_format='{desc}{bar}{r_bar}')
    for batch_idx, (inputs, targets) in t:
        # clf_optimizer.zero_grad()
        inputs = inputs.to(device)
        # inputs, targets = inputs.to(device), targets.to(device)
        # representation = net(inputs).detach()
        representation = net(inputs)
        features += [representation.cpu()]

        # features.append(representation.cpu().numpy())

        # predictions = clf(representation)
        # loss = criterion(predictions, targets)
        # print("Parameters:", list(clf.parameters())) # TO DEBUG
        # loss.requires_grad = True # ADDED 
        # loss.backward()
        # clf_optimizer.step()

        # train_loss += loss.item()

        # t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))

        # acc = test(testloader, device, net, clf)
        # print(f"Accuracy: {acc:.3f}%")

    features = torch.cat(features,dim=0)
    
    save_to = '/home/eecs/tiffany_ding/data/simclr_representations/imagenet_val.pt'
    torch.save(features,save_to)
    #with open(save_to, 'wb') as f:
    #    np.save(f, features)
    #    print('Saved SimCLR representations to ', save_to)
    # Save trained classifier weights
    # save_to = f'trained_classifiers/val-all_epochs={args.num_epochs}.pt'
    # torch.save(clf.state_dict(), save_to)
    # print(f'Saved classifier weights to {save_to}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimCLR verifier')
    parser.add_argument('pth_path', type=str, help='path of the input checkpoint file')
    # parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    # parser.add_argument("--momentum", default=0.9, type=float, help='SGD momentum')
    # parser.add_argument("--batch-size", type=int, default=512, help='Training batch size')
    # parser.add_argument("--num-epochs", type=int, default=90, help='Number of training epochs')
    # parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
    # parser.add_argument("--weight-decay", type=float, default=1e-6, help='Weight decay on the linear classifier')
    # parser.add_argument("--nesterov", action="store_true", help="Turn on Nesterov style momentum")
    args = parser.parse_args()
    run(args.pth_path)


#####


