#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
from datetime import datetime,timedelta  
import sagemaker
import time
import argparse
import os
import sys 
import logging
import torch.utils.data.distributed
import torch.nn.functional as F
import torch.distributed as dist

#TODO: Import dependencies for Debugging andd Profiling

from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

bucket = 'sagemaker-us-east-1-596173457496'

def test(model, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    hook = get_hook(create_if_not_exists=True)
    
    hook.set_mode(modes.EVAL)

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

    
def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    data_loader = torch.utils.data.DataLoader(data, batch_size)
    return data_loader

    

def train(model, train_loader, valid_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook = get_hook(create_if_not_exists=True)
    hook.set_mode(modes.TRAIN)
    batch_size = args.batch_size
    lr = args.lr
    epoch = args.epochs
    lowest_accuracy = 0.000001   #cut-off for stopping training 
    old_val_acc = 0 # initializing the validation accuracy for later comparison 
    
    if hook:
        hook.register_loss(criterion)
    model.train()  # should we remove this?
    for e in range(epoch):
        print("START TRAINING")
        if hook:
            hook.set_mode(modes.TRAIN)
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
        if hook:
            hook.set_mode(modes.EVAL)
        model.eval()
        val_loss = 0
        val_correct = 0
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



        print(f"Epoch {e}: Train Loss {running_loss/len(train_loader.dataset)},Accuracy={100*(correct/len(train_loader.dataset))}, Validation loss               {val_loss/len(valid_loader)}, Val_Accuracy={100*(val_correct/len(valid_loader))}  batch size: {batch_size}  -- learning rate: {lr}")
        
        new_val_acc = 100*(val_correct/len(valid_loader))
        
        acc_diff = new_val_acc - old_val_acc 
                           
        if acc_diff < lowest_accuracy: 
            print('no improvement in accuracy')
            break
        
        old_val_acc = new_val_acc
        #TODO: Finish the rest of the training code
        # The code should stop training when the validation accuracy
        # stops increasing
    return model


    
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



# def create_data_loaders(data, batch_size):
#     '''
#     This is an optional function that you may or may not need to implement
#     depending on whether you need to use data loaders or not
#     '''
    # trainset = 'dogImages/train'

    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #         shuffle=True)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #         download=True, transform=testing_transform)

    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #         shuffle=False)

# https://github.com/johnmburt/projects/blob/master/AWS/sagemaker_dog_breed_id/dog_breed_classifier_gen_LST.ipynb 

# I have copied the code from this repo for lst file creation


def main(args):

    # I think I need to add the hyperparameters here as argparser 
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    # hook = smd.Hook.create_from_json_file()
    # hook.register_hook(model)
    
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    
    #hamza -- how to use this transform on dataset u get from s3?
            
    # dir containing image files
    # NOTE: code assumes this script is run from directory 
    #  containing srcdir.

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
#     path = bucket 
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

#     torch.save(model, path)
#     torch.save(model.state_dict(), "resnet_dog.pt")

if __name__=='__main__':
    
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
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

    #HAMZA # do u need to pass kwargs to data loaders?
    
#     train_kwargs = {"batch_size": args.batch_size}
#     test_kwargs = {"batch_size": args.test_batch_size}
        
    main(args)



