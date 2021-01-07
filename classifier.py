import torch
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl


class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 10)
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss)
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer


if __name__ == '__main__':
    import argparse
    from torchvision import transforms
    from torchvision.datasets.mnist import MNIST
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_epochs', default=30, type=int)
    parser.add_argument('--n_gpus', default=0, type=int)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # dataloaders
    train_dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # model
    model = Classifier()

    # training
    trainer = pl.Trainer(max_epochs=args.n_epochs, gpus=args.n_gpus)
    trainer.fit(model, train_loader)

    # testing
    result = trainer.test(model, test_dataloaders=test_loader)
    print(result)