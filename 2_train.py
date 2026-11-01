import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings('ignore')

class GPRDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, img_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        self.transform_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.transform_eval = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        self.train_ds = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), self.transform_train)
        self.val_ds = datasets.ImageFolder(os.path.join(self.data_dir, 'val'), self.transform_eval)
        self.test_ds = datasets.ImageFolder(os.path.join(self.data_dir, 'test'), self.transform_eval)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

class GPRClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        
        # Instantiate base architecture
        self.model = timm.create_model(cfg.model.architecture, pretrained=True)
        
        # [CRITICAL FIX] Explicitly define the stochastic classification head.
        # This mathematically guarantees the presence of nn.Dropout for Monte Carlo sampling.
        num_ftrs = self.model.get_classifier().in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=cfg.model.dropout_rate),
            nn.Linear(num_ftrs, 2)
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.model.label_smoothing)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), 
            lr=self.cfg.model.learning_rate, 
            weight_decay=self.cfg.model.weight_decay
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

def main():
    print("[INFO] Initializing PyTorch Lightning Training Sequence...")
    cfg = OmegaConf.load("config.yaml")
    
    data_module = GPRDataModule(
        data_dir=cfg.paths.processed_data, 
        batch_size=cfg.data.batch_size, 
        img_size=cfg.data.img_size
    )
    
    model = GPRClassifier(cfg)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir, 
        filename='best_gpr_model',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=cfg.training.patience,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='auto',
        devices=1,
        enable_progress_bar=True
    )

    trainer.fit(model, datamodule=data_module)
    print(f"[SUCCESS] Training completed. Best weights saved to {cfg.paths.checkpoint_dir}")

if __name__ == "__main__":
    main()