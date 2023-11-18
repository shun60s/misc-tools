# -*- coding: UTF-8 -*-
"""Training procedure for NICE.
"""
# There are some changes from original NICE-master's code. 
# This is CPU  beside original uses CUDA.
# Please see LICENSE-NICE-master.txt about the original license.
#
# Check version:
#    Python 3.6.4 on win32
#    torch  1.7.1+cpu
#    torchvision 0.8.2+cpu
#    numpy 1.19.5


import argparse
import torch, torchvision
import numpy as np
import nice, utils

def main(args):
    ###device = torch.device("cuda:0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model hyperparameters
    dataset = args.dataset
    batch_size = args.batch_size
    latent = args.latent
    max_iter = args.max_iter
    sample_size = args.sample_size
    coupling = args.coupling
    mask_config = 1.

    # optimization hyperparameters
    lr = args.lr
    momentum = args.momentum
    decay = args.decay

    zca = None
    mean = None
    if dataset == 'mnist':
        mean = torch.load('./statistics/mnist_mean.pt')
        (full_dim, mid_dim, hidden) = (1 * 28 * 28, 1000, 5)
        transform = torchvision.transforms.ToTensor()
        """
        trainset = torchvision.datasets.MNIST(root='~/torch/data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=batch_size, shuffle=True, num_workers=2)
        """
        trainset = torchvision.datasets.MNIST(root='./data',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=batch_size, shuffle=True, num_workers=0)  # change num_workers
        
    elif dataset == 'fashion-mnist':
        mean = torch.load('./statistics/fashion_mnist_mean.pt')
        (full_dim, mid_dim, hidden) = (1 * 28 * 28, 1000, 5)
        transform = torchvision.transforms.ToTensor()
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=batch_size, shuffle=True, num_workers=2)
    elif dataset == 'svhn':
        zca = torch.load('./statistics/svhn_zca_3.pt')
        mean = torch.load('./statistics/svhn_mean.pt')
        (full_dim, mid_dim, hidden) = (3 * 32 * 32, 2000, 4)
        transform = torchvision.transforms.ToTensor()
        trainset = torchvision.datasets.SVHN(root='~/torch/data/SVHN',
            split='train', download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=batch_size, shuffle=True, num_workers=2)
    elif dataset == 'cifar10':
        zca = torch.load('./statistics/cifar10_zca_3.pt')
        mean = torch.load('./statistics/cifar10_mean.pt')
        transform = torchvision.transforms.Compose(
        [torchvision.transforms.RandomHorizontalFlip(p=0.5),
         torchvision.torchvisitransforms.ToTensor()])
        (full_dim, mid_dim, hidden) = (3 * 32 * 32, 2000, 4)
        trainset = torchvision.datasets.CIFAR10(root='~/torch/data/CIFAR10',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=batch_size, shuffle=True, num_workers=2)
     
    if latent == 'normal':
        prior = torch.distributions.Normal(
            torch.tensor(0.).to(device), torch.tensor(1.).to(device))
    elif latent == 'logistic':
        prior = utils.StandardLogistic()

    filename = '%s_' % dataset \
             + 'bs%d_' % batch_size \
             + '%s_' % latent \
             + 'cp%d_' % coupling \
             + 'md%d_' % mid_dim \
             + 'hd%d_' % hidden

    flow = nice.NICE(prior=prior, 
                coupling=coupling, 
                in_out_dim=full_dim, 
                mid_dim=mid_dim, 
                hidden=hidden, 
                mask_config=mask_config).to(device)
    optimizer = torch.optim.Adam(
        flow.parameters(), lr=lr, betas=(momentum, decay), eps=1e-4)

    total_iter = 0
    train = True
    running_loss = 0

    while train:
        for _, data in enumerate(trainloader, 1):
            flow.train()    # set to training mode
            if total_iter == max_iter:
                train = False
                break

            print ('total_iter ', total_iter ) # add

            total_iter += 1
            optimizer.zero_grad()    # clear gradient tensors

            inputs, _ = data
            inputs = utils.prepare_data(
                inputs, dataset, zca=zca, mean=mean).to(device)

            # log-likelihood of input minibatch
            loss = -flow(inputs).mean()
            running_loss += float(loss)

            # backprop and update parameters
            loss.backward()
            optimizer.step()

            if total_iter % 1000 == 0:
                mean_loss = running_loss / 1000
                bit_per_dim = (mean_loss + np.log(256.) * full_dim) \
                            / (full_dim * np.log(2.))
                print('iter %s:' % total_iter, 
                    'loss = %.3f' % mean_loss, 
                    'bits/dim = %.3f' % bit_per_dim)
                running_loss = 0.0

                flow.eval()        # set to inference mode
                with torch.no_grad():
                    z, _ = flow.f(inputs) # Transformation f: X -> Z
                    reconst = flow.g(z).cpu() #Transformation g: Z -> X (inverse of f).
                    
                    reconst = utils.prepare_data(
                        reconst, dataset, zca=zca, mean=mean, reverse=True)
                    
                    
                    samples = flow.sample(sample_size).cpu()
                    samples = utils.prepare_data(
                        samples, dataset, zca=zca, mean=mean, reverse=True)
                    
                    
                    torchvision.utils.save_image(torchvision.utils.make_grid(reconst),
                        './reconstruction/' + filename +'iter%d.png' % total_iter)
                    torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                        './samples/' + filename +'iter%d.png' % total_iter)

    print('Finished training!')

    torch.save({
        'total_iter': total_iter, 
        'model_state_dict': flow.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(), 
        'dataset': dataset, 
        'batch_size': batch_size, 
        'latent': latent, 
        'coupling': coupling, 
        'mid_dim': mid_dim, 
        'hidden': hidden, 
        'mask_config': mask_config}, 
        './models/mnist/' + filename +'iter%d.tar' % total_iter)

    print('Checkpoint Saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MNIST NICE PyTorch implementation')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=200)
    parser.add_argument('--latent',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--max_iter',
                        help='maximum number of iterations.',
                        type=int,
                        default=25000)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)
    parser.add_argument('--momentum',
                        help='beta1 in Adam optimizer.',
                        type=float,
                        default=0.9)
    parser.add_argument('--decay',
                        help='beta2 in Adam optimizer.',
                        type=float,
                        default=0.999)
    parser.add_argument('--coupling',
                        help='number of coupling.',
                        type=int,
                        default=4)
    args = parser.parse_args()
    main(args)
