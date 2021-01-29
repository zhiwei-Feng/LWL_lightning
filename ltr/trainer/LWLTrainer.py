import torch
from pytorch_lightning.core.lightning import LightningModule


class LitLWL(LightningModule):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
