import argparse
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from auto_encoder import AutoEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--latent_dim', default=64, type=int)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

#
pl.seed_everything(args.seed)

# dataloaders
train_dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
test_dataset = MNIST('', train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)