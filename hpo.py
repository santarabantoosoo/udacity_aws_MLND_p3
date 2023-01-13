#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import sagemaker
from sagemaker.pytorch import PyTorch
import os
import torch.utils.data.distributed
import torch.nn.functional as F
import torch.distributed as dist

from sagemaker.session import TrainingInput

# sagemaker_session = sagemaker.Session()

# bucket = sagemaker_session.default_bucket()

bucket = 'sagemaker-us-east-1-596173457496'

import json
import logging
import os
import sys
import argparse


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# sagemaker_session = sagemaker.Session()

# bucket = sagemaker_session.default_bucket()

from sagemaker.session import TrainingInput


def test(model, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader)
    total_acc = running_corrects / len(test_loader)


def train(model, train_loader, valid_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    batch_size = args.batch_size
    epoch = args.epochs

    
    model.train()  # should we remove this?
    for e in range(epoch):
        print("START TRAINING")
        running_loss=0
        correct=0
        for data, target in train_loader:
            # data=data.to(device)
            # target=target.to(device)
            optimizer.zero_grad()
            pred = model(data)             #No need to reshape data since CNNs take image inputs
            loss = criterion(pred, target)
            running_loss+=loss
            loss.backward()
            optimizer.step()
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
        print("START VALIDATING")
        model.eval()
        val_loss = 0
        val_correct = 0
        val_accuracy = []
        with torch.no_grad():
            for data, target in valid_loader :
                # inputs, targets = inputs.to(device), targets.to(device)
                pred = model(data)
                loss = criterion(pred, target)
                val_loss += loss
                pred=pred.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
        
#         epoch_time = time.time() - start
#         epoch_times.append(epoch_time)
        
        print(f"Epoch {e}: Train Loss {running_loss/len(train_loader.dataset)},Accuracy={100*(correct/len(train_loader.dataset))}, Validation loss               {val_loss/len(valid_loader)}, Val_Accuracy={100*(val_correct/len(valid_loader))}")
        # hamza may need to implement train stop if the below condition is reached 
        #TODO: Finish the rest of the training code
        # The code should stop training when the validation accuracy
        # stops increasing
    return model

# patience = 5 # stop if the network stop improving for 5 epochs
# delta_loss = 1e-4 # less than this, the network is not improving
# best_loss = 99
# current_tol = 0
# for _ in range(epochs):
#     ...
#     if eval_flag: 
#         val_loss = ... # new loss from training network
#         if val_loss - best_loss > delta_loss:
#             best_loss = val_loss
#             cur_patience = 0
#         else: 
#             cur_patience += 1
        
#         if cur_patience >= patience:
#             break



def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    data_loader = torch.utils.data.DataLoader(data, batch_size)
    return data_loader

def net():
    model = models.resnet50(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential( nn.Linear( num_features, 128), 
                             nn.ReLU(inplace = True),
                             nn.Linear(128, 133),
                             nn.ReLU(inplace = True) 
                            )
    return model


              
              
def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    # train_path = 's3://sagemaker-us-east-1-596173457496/project3/train/'
    # valid_path = 's3://sagemaker-us-east-1-596173457496/project3/valid/'
    # test_path = 's3://sagemaker-us-east-1-596173457496/project3/test/'

    train_path = args.data_dir + "/train"
    valid_path = args.data_dir + "/valid"
    test_path = args.data_dir + "/test"
    
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_data = torchvision.datasets.ImageFolder(root = train_path, transform = transform)
    valid_data = torchvision.datasets.ImageFolder(root = valid_path, transform = transform)
    test_data = torchvision.datasets.ImageFolder(root = test_path, transform = transform)
    
    train_loader = create_data_loaders(train_data, args.batch_size)
    
    valid_loader = create_data_loaders(valid_data, args.batch_size)
    
    test_loader = create_data_loaders(test_data, args.batch_size)
    
    
    model=train(model, train_loader, valid_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''

    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

              

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, metavar="LR", help="learning rate (default: 0.1)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train",
    )
    
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
 
    args = parser.parse_args()
        
    main(args)






