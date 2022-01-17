'''
python3 main.py --ds-folder "../datasets/" --datasource MNIST --lr 1e-4 --batch-size 16 --num-classes 10 --logdir /media/n10/Data/ -epsilon -0.1 --num-epochs 10 --resume-epoch 0 --train
'''

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision

import numpy as np

import os
import typing
import argparse

from ConvNet import ConvNet
from utils import grad_estimation

parser = argparse.ArgumentParser(description='Setup variables')

parser.add_argument('--ds-folder', type=str, default='../datasets', help='Parent folder containing the dataset')
parser.add_argument('--datasource', type=str, help='Name of dataset')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for meta-update')
parser.add_argument('--batch-size', type=int, default=16, help='Minibatch of episodes to update meta-parameters')

parser.add_argument('--num-classes', type=int, default=10, help='Number of classes')

# parser.add_argument('--img-size', action='append', help='A pair of image size: 32 or 84')

parser.add_argument('--logdir', type=str, default='/media/n10/Data/', help='Folder to store model and logs')

parser.add_argument('--num-epochs', type=int, default=1, help='')
parser.add_argument('--resume-epoch', type=int, default=0, help='Resume')

parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.set_defaults(train_flag=True)

parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon')

args = parser.parse_args()
print()

config = {}
for key in args.__dict__:
    config[key] = args.__dict__[key]

config['device'] = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
config['train'] = True

# config['datasource'] = 'MNIST'
# config['num_classes'] = 10
# config['batch_size'] = 32

config['logdir'] = os.path.join(
    config['logdir'],
    'direct_loss_minimization',
    config['datasource'],
    'loss_01'
)
if not os.path.exists(path=config['logdir']):
    from pathlib import Path
    Path(config['logdir']).mkdir(parents=True, exist_ok=True)

# config['resume_epoch'] = 0
# config['lr'] = 1e-4

# prepare data
mnist_dataset_train = torchvision.datasets.MNIST(
    root='../datasets/',
    train=True,
    transform=torchvision.transforms.ToTensor()
)
mnist_dataset_test = torchvision.datasets.MNIST(
    root='../datasets/',
    train=False,
    transform=torchvision.transforms.ToTensor()
)

data_loader_train = torch.utils.data.DataLoader(
    dataset=mnist_dataset_train,
    batch_size=config['batch_size'],
    drop_last=True
)
data_loader_test = torch.utils.data.DataLoader(
    dataset=mnist_dataset_test,
    batch_size=config['batch_size'],
    drop_last=False
)

# prepare model
net = ConvNet(dim_output=config['num_classes'])

# pass one data point through to initialize lazy layers
for x, _ in data_loader_train:
    net.forward(input=x)
    break

optimizer = torch.optim.Adam(params=net.parameters(), lr=config['lr'])

if (config['resume_epoch'] > 0):
    checkpoint_filename = 'Epoch_{0:d}.pt'.format(config['resume_epoch'])
    checkpoint_fullpath = os.path.join(config['logdir'], checkpoint_filename)
    if torch.cuda.is_available():
        saved_checkpoint = torch.load(
            checkpoint_fullpath,
            map_location=lambda storage,
            loc: storage.cuda(0)
        )
    else:
        saved_checkpoint = torch.load(
            checkpoint_fullpath,
            map_location=lambda storage,
            loc: storage
        )

    net.load_state_dict(state_dict=saved_checkpoint['net_state_dict'])
    optimizer.load_state_dict(state_dict=saved_checkpoint['optimizer_state_dict'])

def train() -> None:
    try:
        # tensorboard to monitor
        tb_writer = SummaryWriter(
            log_dir=config['logdir'],
            purge_step=config['resume_epoch'] if config['resume_epoch'] > 0 else None
        )

        for epoch in range(config['resume_epoch'], config['resume_epoch'] + config['num_epochs']):
            monitor = {
                'loss': 0.,
                'batch_count': 0
            }

            for x, y in data_loader_train:
                loss, grad = grad_estimation(x=x, y=y, net=net, num_classes=config['num_classes'], epsilon=config['epsilon'])
                for i, p in enumerate(net.parameters()):
                    p.grad = grad[i]
                optimizer.step()
                optimizer.zero_grad()

                monitor['loss'] += loss.item()
                monitor['batch_count'] += 1

            # add training loss to tensorboard
            tb_writer.add_scalar(
                tag='Loss',
                scalar_value=monitor['loss'] / monitor['batch_count'],
                global_step=epoch + 1
            )

            # validation
            val_results = test(net=net, data_loader=data_loader_test)
            for key in val_results:
                tb_writer.add_scalar(
                    tag='Val/{0:s}'.format(key),
                    scalar_value=val_results[key],
                    global_step=epoch + 1
                )

            # --------------------------------------------------
            # save model
            # --------------------------------------------------
            checkpoint = {
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            checkpoint_filename = 'Epoch_{0:d}.pt'.format(epoch + 1)
            torch.save(checkpoint, os.path.join(config['logdir'], checkpoint_filename))
            checkpoint = 0
            print('SAVING parameters into {0:s}'.format(checkpoint_filename))
            print('----------------------------------------\n')
    finally:
        # --------------------------------------------------
        # clean up
        # --------------------------------------------------
        print('\nClose tensorboard summary writer')
        tb_writer.close()
    
    return None

def test(net: torch.nn.Module, data_loader: torch.utils.data.DataLoader) -> typing.Dict[str, float]:
    monitor = {
        'NLL': 0.,
        'accuracy': 0.
    }
    for x, y in data_loader:
        logits = net.forward(input=x)
        monitor['NLL'] += torch.nn.functional.cross_entropy(
            input=logits,
            target=y,
            reduction='sum'
        )
        monitor['accuracy'] += (logits.argmax(dim=1) == y).float().sum().item()
    
    for key in monitor:
        monitor[key] /= len(data_loader.dataset)
    
    return monitor

def main():
    """
    """
    if config['train']:
        train()
    else:
        val_results = test(net=net, data_loader=data_loader_test)
        print(val_results)

if __name__ == '__main__':
    main()