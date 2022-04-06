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


device = 'cuda' # Other option: 'cpu'


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
            logits = clf(features)
            clf_loss = criterion(logits, targets)
        
        
            test_clf_loss += clf_loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            t.set_description('Loss: %.3f | Test Acc: %.3f%% ' % (test_clf_loss / (batch_idx + 1), 100. * correct / total))

    acc = 100. * correct / total
    return acc

def get_train_and_val_dataloaders(batch_size=64):
    '''
    Returns 2 DataLoaders, one containing all of ImageNet-train and another containing all of ImageNet-val
    '''
    dataloader_list = []
    for dataset_name in ['train', 'val']:
        print(f'Loading SimCLR representations for ImageNet {dataset_name}...')
        representation_location = f'/home/eecs/tiffany_ding/code/SimCLRv2-Pytorch/.cache/simclr_representations/imagenet_{dataset_name}'
        features = torch.load(representation_location+'_features.pt')
        labels = torch.load(representation_location+'_labels.pt')
        # breakpoint()
        print('Dimension of features:', features.shape)
        dataset = torch.utils.data.TensorDataset(features,labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
        dataloader_list.append(dataloader)

    return dataloader_list[0], dataloader_list[1]

def get_split_dataloaders(dataset_name, train_split, test_split='default', calib_only=False):
    '''
    Load in either ImageNet train or val representations and split the dataset into two
    '''
    
    representation_location = f'/home/eecs/tiffany_ding/code/SimCLRv2-Pytorch/.cache/simclr_representations/imagenet_{dataset_name}'
    #dataset_name = 'train' # 'train' or 'val'
    #train_split = 0.7  # what fraction of data to use for training 
    
    features = torch.load(representation_location+'_features.pt')
    labels = torch.load(representation_location+'_labels.pt')
        
    clfdataset = torch.utils.data.TensorDataset(features,labels)
    train_size = int(len(clfdataset) * train_split)
    if test_split == 'default': # Everything that is not in train will be included in test
        test_size = len(clfdataset) - train_size
        calib_size = 0
        splits = [train_size, test_size, calib_size]
        train_dataset, test_dataset, calibration_dataset = torch.utils.data.random_split(clfdataset, splits, generator=torch.Generator().manual_seed(0))
    else:
        test_size = int(len(clfdataset) * test_split)
        splits = [train_size, test_size, len(clfdataset) - (train_size + test_size)]
        train_dataset, test_dataset, calibration_dataset = torch.utils.data.random_split(clfdataset, splits, generator=torch.Generator().manual_seed(0))
        
    if calib_only:
        clftrainloader = None
        testloader = None
    else:
        clftrainloader = DataLoader(train_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=0)
        testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=0)
        
    calibloader = DataLoader(calibration_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=0)

    print(f'Size of classifier training set: {train_size}')
    print(f'Size of classifier test set: {test_size}')
    print(f'Size of calibration/evaluation set: {len(calibration_dataset)}')
    return clftrainloader, testloader, calibloader


def get_logits(save_prefix, num_epochs=90, weights_path=None):
    
    clf_train_split = 0.1 # What fraction of ImageNet train we should use to train the classifier
    clf_test_split = 0.01 # What fraction of ImageNet train we should use to compute classifier accuracy
    
    if weights_path is None:
        # Train classifier
        
        clftrainloader, clftestloader, calib_loader = get_split_dataloaders('train', train_split=clf_train_split, test_split=clf_test_split)
        weights_path = '.cache/trained_classifiers/train-0.1.pt'
        clf = train(clftrainloader, clftestloader, weights_path, num_epochs, learning_rate=.001)  
    else:
        # Load classifier weights
        simclr_feature_dim = 6144
        num_classes = 1000
        clf = torch.nn.Linear(simclr_feature_dim, num_classes)
        clf.load_state_dict(torch.load(weights_path))
        clf.to(device)
        
        # We only compute logits for the data we haven't already used 
        _, _, calib_loader = get_split_dataloaders('train', train_split=clf_train_split, test_split=clf_test_split, calib_only=True)
        
        
    with torch.no_grad():
        logits = []
        labels = []
        
        print('Computing logits...')
        t = tqdm(enumerate(calib_loader), total=len(calib_loader), desc='Batch:')
        for batch_idx, (features, targets) in t:
            features, targets = features.to(device), targets.to(device)
            curr_logits = clf(features)
            
            logits += [curr_logits]
            labels += [targets]

        # Concatenate
        logits = torch.cat(logits,dim=0)
        labels = torch.cat(labels,dim=0)
        
        
    # Save logits
    torch.save(logits,save_prefix + '_logits.pt')
    torch.save(labels,save_prefix + '_labels.pt')
    print(f'Saved logits to', save_prefix + '_logits.pt')
    print(f'Saved labels to', save_prefix + '_labels.pt')
    
    
    
def train(clftrainloader, clftestloader, save_to, num_epochs, learning_rate=.001):

    print(f'After training, weights will be saved to {save_to}')

    simclr_feature_dim = 6144
    num_classes = 1000
    clf = torch.nn.Linear(simclr_feature_dim, num_classes).to(device)
    clf.train()
    
    criterion = torch.nn.CrossEntropyLoss()
    clf_optimizer = torch.optim.Adam(clf.parameters(), lr=learning_rate, weight_decay=1e-6)

    print(f'Training for {num_epochs} epochs')
    for epoch in range(num_epochs):
        print('Epoch', epoch)
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

        acc = test(clftestloader, device, clf)
        print(f"Accuracy: {acc:.3f}%")

    # Save trained classifier weights
    save_to = save_to + f'acc={acc / 100:.4f}.pt'
    torch.save(clf.state_dict(), save_to)
    print(f'Saved classifier weights to {save_to}')
    
    return clf
 

def run(args):
    
    # Set location to save weights
    save_prefix = 'train-val' # UPDATE THIS AS NECESSARY
    #save_prefix = 'train-0.7'
    save_to = f'.cache/trained_classifiers/{save_prefix}_epochs={args.num_epochs}'
    print(f'After training, weights will be saved to {save_to}[...]')
    
    # OPTION 1: Train classifier and save weights
    # Load data
#     clftrainloader, clftestloader, _ = get_split_dataloaders("train", train_split=0.7)
    clftrainloader, clftestloader = get_train_and_val_dataloaders(batch_size=64)
    train(clftrainloader, clftestloader, save_to, args.num_epochs, learning_rate=.01)
    

    # OPTION 2: Train classifier and apply classifier to get logits for data not used to train
    #save_prefix = f'.cache/logits/imagenet_train_subset'
    #get_logits(save_prefix, num_epochs=args.num_epochs, weights_path=None)
   
    # OPTION 3: Load pretrained classifier weights and apply classifier for data not used to train
#     save_prefix = f'.cache/logits/imagenet_train_subset'
#     weights_path = f'.cache/trained_classifiers/train-all_epochs=10.pt'
#     get_logits(save_prefix, weights_path=weights_path)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train downstream classifier with gradients.')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs')
    run(parser.parse_args())

