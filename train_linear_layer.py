import os
import argparse
from collections import Counter

import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from resnet import get_resnet, name_to_params
import pdb


def test(testloader, device, clf):
    criterion = torch.nn.CrossEntropyLoss()
    clf.eval()
    test_clf_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        t = tqdm(enumerate(testloader), total=len(testloader), desc='Loss: **** | Test Acc: ****% ',
                 bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (features, targets) in t:
            features, targets = features.to(device), targets.to(device)
            probabilities = clf(features)
            clf_loss = criterion(probabilities, targets)
        
        
            test_clf_loss += clf_loss.item()
            _, predicted = probabilities.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            t.set_description('Loss: %.3f | Test Acc: %.3f%% ' % (test_clf_loss / (batch_idx + 1), 100. * correct / total))

    acc = 100. * correct / total
    return acc



def run(args):
    device = 'cuda' # 'cpu'

    # Load data
    representation_location = '/home/eecs/tiffany_ding/code/SimCLRv2-Pytorch/.cache/simclr_representations/imagenet_val'
    # pdb.set_trace()
    features = torch.load(representation_location+'_features.pt')
    labels = torch.load(representation_location+'_labels.pt')
    clfdataset = torch.utils.data.TensorDataset(features,labels)
    clftrainloader = DataLoader(clfdataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=0)
    testloader = DataLoader(clfdataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=0) # NOTE: currently testing on the same dataset we are training on

    num_classes = 1000
    clf = torch.nn.Linear(features.shape[1], num_classes).to(device)
    clf.train()
    
    criterion = torch.nn.CrossEntropyLoss()
    clf_optimizer = torch.optim.Adam(clf.parameters(), lr=1e-4)

    for epoch in range(args.num_epochs):
        train_loss = 0
        t = tqdm(enumerate(clftrainloader), desc='Loss: **** ', total=len(clftrainloader), bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (features, targets) in t:
            clf_optimizer.zero_grad()
            features, targets = features.to(device), targets.to(device)
            predictions = clf(features)
            loss = criterion(predictions, targets)
            loss.backward()
            clf_optimizer.step()

            train_loss += loss.item()

            t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))

        acc = test(testloader, device, clf)
        print(f"Accuracy: {acc:.3f}%")

    # Save trained classifier weights
    save_to = f'.cache/trained_classifiers/val-all_epochs={args.num_epochs}.pt'
    torch.save(clf.state_dict(), save_to)
    print(f'Saved classifier weights to {save_to}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train downstream classifier with gradients.')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs')
    run(parser.parse_args())

