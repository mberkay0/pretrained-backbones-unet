from backbones_unet.model.unet import Unet
import torch
from backbones_unet.utils.reproducibility import set_seed
from backbones_unet.utils.dataset import SemanticSegmentationDataset
from backbones_unet.model.losses import DiceLoss
from backbones_unet.utils.trainer import Trainer

# set_seed()

random_tensor = torch.rand((1, 3, 64, 64))

model = Unet(in_channels=3, num_classes=1)

print(model.predict(random_tensor))