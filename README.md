<div align="center">
<h1>
  Pretrained Backbones with UNet
</h1>
<div>

**A [PyTorch](https://pytorch.org/)-based Python library with UNet architecture and multiple backbones for Image Semantic Segmentation.**


[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/qubvel/segmentation_models.pytorch/blob/master/LICENSE) 
[![PyPI](https://img.shields.io/pypi/v/pretrained-backbones-unet?color=blue&style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/pretrained-backbones-unet/) 
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pretrained-backbones-unet?style=for-the-badge&color=blue)](https://pepy.tech/project/pretrained-backbones-unet) 
<br>
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.9+-red?style=for-the-badge&logo=pytorch)](https://pepy.tech/project/segmentation-models-pytorch) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&logo=python&logoColor=white)](https://pepy.tech/project/segmentation-models-pytorch) 
</div>
</div>

## <div align="center">Overview</div>

This is a simple package for semantic segmentation with [UNet](https://arxiv.org/pdf/1505.04597.pdf) and pretrained backbones. This package utilizes the [timm models](https://pypi.org/project/timm/) for the pre-trained encoders.

When dealing with relatively limited datasets, initializing a model using pre-trained weights from a large dataset can be an excellent choice for ensuring successful network training. By utilizing state-of-the-art models, such as ConvNeXt, as an encoder, you can effortlessly solve the problem at hand while achieving optimal performance in this context.

The primary characteristics of this library are as follows:
*   430 pre-trained backbone networks are available for the UNet semantic segmentation model.

*   Supports backbone networks such as ConvNext, ResNet, EfficientNet, DenseNet, RegNet, and VGG... which are popular and SOTA performers, for the UNet model.

*   It is possible to adjust which layers of the backbone of the model are trainable parametrically.

*   It includes a DataSet class for binary and multi-class semantic segmentation.

*   And it comes with a pre-built rapid custom training class.
## Installation

### Pypi version:
```
pip install pretrained-backbones-unet
```

### Source code version:
```
pip install git+https://github.com/mberkay0/pretrained-backbones-unet
```

## Usage
```python
from backbones_unet.model.unet import Unet
from backbones_unet.utils.dataset import SemanticSegmentationDataset
from backbones_unet.model.losses import DiceLoss
from backbones_unet.utils.trainer import Trainer

# create a torch.utils.data.Dataset/DataLoader
train_img_path = 'example_data/train/images' 
train_mask_path = 'example_data/train/masks'

val_img_path = 'example_data/val/images' 
val_mask_path = 'example_data/val/masks'

train_dataset = SemanticSegmentationDataset(train_img_path, train_mask_path)
val_dataset = SemanticSegmentationDataset(val_img_path, val_mask_path)

train_loader = DataLoader(train_dataset, batch_size=2)
val_loader = DataLoader(val_dataset, batch_size=2)

model = Unet(
    backbone='convnext_base', # backbone network name
    in_channels=3,            # input channels (1 for gray-scale images, 3 for RGB, etc.)
    num_classes=1,            # output channels (number of classes in your dataset)
)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, 1e-4) 

trainer = Trainer(
    model,                    # UNet model with pretrained backbone
    criterion=DiceLoss(),     # loss function for model convergence
    optimizer=optimizer,      # optimizer for regularization
    epochs=10                 # number of epochs for model training
)

trainer.fit(train_loader, val_loader)
```

## Available Pretrained Backbones
```python
import backbones_unet

print(backbones_unet.__available_models__)
```
