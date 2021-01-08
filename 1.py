import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM


class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = FM.accuracy(logits, y)
        loss = F.cross_entropy(logits, y)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = FM.accuracy(logits, y)
        loss = F.cross_entropy(logits, y)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer


if __name__ == '__main__':
    import argparse
    from torchvision import transforms
    from torchvision.datasets.mnist import MNIST
    from torch.utils.data import DataLoader, random_split

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_epochs', default=30, type=int)
    parser.add_argument('--n_gpus', default=0, type=int)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # dataloaders
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    train_dataset, val_dataset = random_split(dataset, [55000, 5000])
    test_dataset = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # model
    model = Classifier()

    # training
    trainer = pl.Trainer(max_epochs=args.n_epochs, gpus=args.n_gpus)
    trainer.fit(model, train_loader, val_loader)

    # testing
    trainer.test(test_dataloaders=test_loader)
