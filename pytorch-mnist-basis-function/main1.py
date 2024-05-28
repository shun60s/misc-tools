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
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 2, 1)
        
        self.lout=128 # number of output layer
        self.conv2 = nn.Conv2d(32, self.lout, 2)
        self.conv3 = nn.Conv2d(self.lout, self.lout, 2)
        self.conv3 = nn.Conv2d(self.lout, self.lout, 2)
        self.conv4 = nn.Conv2d(self.lout, self.lout, 2)
        
        self.define_r3x3()
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
        """
        print ('x4 size', x4.size())
        print ('x4 size[0,0]', x4[0,0].size())
        print(x4[0,0].to('cpu').detach().numpy().copy())
        """
        # compute absolute value of difference
        x5=torch.abs(torch.sub(x4, self.r3x3))
        """
        print ('x5 size', x5.size())
        print ('x5 size[0,0]', x5[0,0].size())
        print(x5[0,0].to('cpu').detach().numpy().copy())
        """
        # compute average
        x6=torch.mean(x5.view(x5.size(0),x5.size(1),x5.size(2),9), dim=3, keepdim=True) # (128,11,1)
        """
        print ('65 size', x6.size())
        print ('x6 size[0,0]', x6[0,0].size())
        print(x6[0,0].to('cpu').detach().numpy().copy())
        """
        
        x7 = torch.flatten(x6, 1) # (1408)
        x = self.fc1(x7)
        
        output = F.log_softmax(x, dim=1)
        
        return output

    def define_r3x3(self):
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
                         [0, 0, 1.0]]], requires_grad=False)


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
    args = parser.parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
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
    
    model = Net().to(device)
    
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


if __name__ == '__main__':
    main()
