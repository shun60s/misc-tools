# -*- coding: UTF-8 -*-

"""

This is a copy from pytorch/exmple/mnist <https://github.com/pytorch/examples/tree/main/mnist> and changes for
MNIST digits classification using 3x3 pixels basis function (PyTorch implement).

"""
"""
This is CPU  beside original uses CUDA.
Check version:
    Python 3.6.4 on win32
    torch  1.7.1+cpu
    torchvision 0.8.2+cpu
    matplotlib 3.3.1
    numpy 1.18.4
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import numpy as np

class Net(nn.Module):
    def __init__(self,device):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 2, 1)
        
        self.lout=128 # number of output layer
        self.conv2 = nn.Conv2d(32, self.lout, 2)
        self.conv3 = nn.Conv2d(self.lout, self.lout, 2)
        self.conv3 = nn.Conv2d(self.lout, self.lout, 2)
        self.conv4 = nn.Conv2d(self.lout, self.lout, 2)
        
        self.define_r3x3(device)
        self.fc1 = nn.Linear(self.lout * self.r3x3.size(0), 10)

    def forward(self, x):
        x = self.conv1(x)                    # (32, 27,27)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2, padding=1) # (32, 14,14)
        x = self.conv2(x)                    # (128, 13,13)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2, padding=1) # (128, 7,7)
        x = self.conv3(x)                    # (128, 6,6)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2, padding=1) # (128, 4,4)
        x3 = self.conv4(x)                   # (128, 3,3)
        x3 = torch.sigmoid(x3)               # use sigmoid to output range in 0-1.0
        
        x4= x3.unsqueeze(2).repeat(1,1,self.r3x3.size(0),1,1) # (128, 11, 3,3)
        
        # compute absolute value of difference
        x5=torch.abs(torch.sub(x4, self.r3x3))
        
        # compute average
        x6=torch.mean(x5.view(x5.size(0),x5.size(1),x5.size(2),9), dim=3, keepdim=True) # (128,11,1)
        
        x7 = torch.flatten(x6, 1) # (1408)
        x = self.fc1(x7)
        
        output = F.log_softmax(x, dim=1)
        
        return output

    def define_r3x3(self,device):
    	# eleven 3x3 pixels basis functions definition
        self.r3x3 = torch.tensor([     \
                        # 1st
                        [[0, 1.0, 0],  \
                         [0, 1.0, 0],  \
                         [0, 1.0, 0]], \
                        # 2nd
                        [[0, 0, 1.0],  \
                         [0, 0, 1.0],  \
                         [0, 0, 1.0]], \
                        # 3rd
                        [[1.0, 0, 0],  \
                         [1.0, 0, 0],  \
                         [1.0, 0, 0]], \
                        # 4th
                        [[0, 0, 0],  \
                         [0, 0, 0],  \
                         [1.0, 1.0, 1.0]], \
                        # 5th
                        [[0, 0, 0],  \
                         [1.0, 1.0, 1.0],  \
                         [0, 0, 0]], \
                        # 6th
                        [[1.0, 1.0, 1.0],  \
                         [0, 0, 0],  \
                         [0, 0, 0]], \
                        # 7th
                        [[1.0, 1.0, 1.0],  \
                         [0, 0, 1.0],  \
                         [1.0, 1.0, 1.0]], \
                        # 8th
                        [[1.0, 1.0, 1.0],  \
                         [1.0, 0, 0],  \
                         [1.0, 1.0, 1.0]], \
                        # 9th
                        [[1.0, 1.0, 1.0],  \
                         [1.0, 0, 1.0],  \
                         [1.0, 1.0, 1.0]], \
                        # 10th
                        [[0, 0, 1.0],  \
                         [0, 1.0, 0],  \
                         [1.0, 0, 0]], \
                        # 11th
                        [[1.0, 0, 0],  \
                         [0, 1.0, 0],  \
                         [0, 0, 1.0]]], requires_grad=False, device=device)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def show_incorrect_image(model, device, test_loader, show_max_number=4):
    # show incorrect prediction image with binarization
    model.eval()
    count=0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            for id, index in enumerate( torch.where(pred.eq(target.view_as(pred)) == False)[0]):
                print ('id ', count)
                print ('true_label', target[index.item()].item(), 'predict_label',pred[index.item()][0].item() )
                print ('value_of_true_label', output[index.item()][target[index.item()].item()].item(), ' value_of_predict_label', output[index.item()][pred[index.item()][0].item()].item() )
                # show incorrect digit binarization image
                for y in range(28):
                    for x in range(28):
                        if  data[index.item()][0][y,x].item() > 0.0:
                            print ( '1', end='')
                        else:
                            print ( '0', end='')
                    print('')
                
                count+=1
                if count >= show_max_number:
                    break
            
            if count >= show_max_number:
                    break


def show_heat_map(model):
	# import matplotlib to draw heat map
    import matplotlib.pyplot as plt
    
    # show heat map of weights of last fully-connected layer, fc1(10, ...
    
    fc1 = model.state_dict()['fc1.weight'].to('cpu').detach().numpy().copy() # get weights of last fully-connected layer, fc1(10, ...
    #print(fc1) # print out weights value
    
    # draw heat map
    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_subplot(1,1,1)
    d10_weights= np.abs(fc1)
    ax.pcolor(d10_weights, cmap=plt.cm.Reds) # set color map to draw heat map
    
    ax.set_title(' heat map of weights of last fully-connected layer')
    ax.set_ylabel('digits')
    ax.set_xlabel('fc1 inputs')
    ax.set_xticks(np.arange(int(fc1.shape[1] / model.lout))* model.lout) # auxiliary scale per one basis function
    ax.set_yticks(np.arange(10))      # auxiliary scale per digit
    plt.tight_layout()
    plt.show()
    

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST digits classification using 3x3 pixels basis function ')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='For Loading the previous Model')
    args = parser.parse_args()  #  = parser.parse_args(args=[])
    use_cuda = args.use_cuda and torch.cuda.is_available()
    
    
    torch.manual_seed(args.seed)
    
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 10,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    model = Net(device).to(device)
    
    model_path ="mnist_cnn.pt"
    if args.load_model:
        model.load_state_dict(torch.load(model_path))
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
    
    if args.save_model:
        torch.save(model.state_dict(), model_path)
        """
        if use_cuda:
            model_path_cpu ="mnist_cnn_cpu.pt"
            import copy
            model2=copy.deepcopy(model)
            torch.save(model2.to('cpu').state_dict(), model_path_cpu)
        """
    if 1: # set 1 to show incorrect prediction image with binarization
        show_incorrect_image(model, device, test_loader)
    
    if 1: # set 1 to show heat map of weights of last fully-connected layer, fc1(10, ...
        show_heat_map(model)


if __name__ == '__main__':
    main()
