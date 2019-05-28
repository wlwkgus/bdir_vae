import argparse
import os
from os import listdir

import imageio
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from data.font_dataset import FontDataset
from mnist.vae import VAE


def make_grid(tensor, number1, number2, size):
    tensor = t.transpose(tensor, 0, 1).contiguous().view(1, number1, number2 * size, size)
    tensor = t.transpose(tensor, 1, 2).contiguous().view(1, number1 * size, number2 * size)

    return tensor


if __name__ == "__main__":

    if not os.path.exists('prior_sampling'):
        os.mkdir('prior_sampling')

    parser = argparse.ArgumentParser(description='CDVAE')
    parser.add_argument('--num-epochs', type=int, default=2000, metavar='NI',
                        help='num epochs (default: 4)')
    parser.add_argument('--batch-size', type=int, default=40, metavar='BS',
                        help='batch size (default: 40)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--save', type=str, default='trained_model', metavar='TS',
                        help='path where save trained model to (default: "trained_model")')
    args = parser.parse_args()

    # dataset = datasets.MNIST(root='data/',
    #                          transform=transforms.Compose([
    #                              transforms.ToTensor()]),
    #                          download=True,
    #                          train=True)
    # dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # transform = transforms.ToTensor()
    print("Generating dataset...")
    data_set = FontDataset('/home/tony/work/font_transform/new_data/data/', train=True)
    dataloader = DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )
    print('Generating dataset done.')

    vae = VAE()
    if args.use_cuda:
        vae = vae.cuda()

    optimizer = Adam(vae.parameters(), args.learning_rate, eps=1e-6)

    likelihood_function = nn.BCEWithLogitsLoss(size_average=False)

    z = [Variable(t.randn(args.batch_size, size)) for size in vae.latent_size]
    if args.use_cuda:
        z = [var.cuda() for var in z]

    for epoch in range(args.num_epochs):
        for iteration, (batch_data) in enumerate(dataloader):

            input = batch_data['base_data'].view(-1, 784) / 255.
            target = batch_data['data'].view(-1, 784) / 255.
            font_vec = batch_data['font']
            char_vec = batch_data['char']
            transform_vec = batch_data['transform']
            # input = Variable(input).view(-1, 784)
            if args.use_cuda:
                input = input.cuda()
                target = target.cuda()
                font_vec = font_vec.cuda()
                char_vec = char_vec.cuda()
                transform_vec = transform_vec.cuda()

            optimizer.zero_grad()

            out, kld = vae(input, char_vec, font_vec, transform_vec)

            target = target.view(-1, 1, 28, 28)
            out = out.contiguous().view(-1, 1, 28, 28)

            likelihood = likelihood_function(out, target) / args.batch_size
            print(likelihood, kld)
            loss = likelihood + kld

            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                print('epoch {}, iteration {}, loss {}'.format(epoch, iteration, loss.cpu().data.numpy().item()))

                sampling = vae.sample(z, char_vec, font_vec, transform_vec).view(-1, 1, 28, 28)

                grid = make_grid(F.sigmoid(sampling).cpu().data, 5, 8, 28)
                vutils.save_image(grid, 'prior_sampling/vae_{}.png'.format(epoch * len(dataloader) + iteration))

    samplings = [f for f in listdir('prior_sampling')]
    samplings = [imageio.imread('prior_sampling/' + path) for path in samplings for _ in range(2)]
    imageio.mimsave('prior_sampling/movie.gif', samplings)

    t.save(vae.cpu().state_dict(), args.save)
