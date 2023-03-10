import torch
import torch.nn as nn
from timm import create_model
import torch.nn.functional as F
from typing import Optional, List
from backbones_unet import __available_models__


class Unet(nn.Module):
    """
    This class utilizes a pre-trained model as a backbone network and 
    rearranges it as an encoder for fully convolutional neural network UNet.
    The UNet model consists of an encoder path and a decoder path. 
    The encoder path is made up of multiple downsampling blocks, 
    which reduce the spatial resolution of the feature maps and 
    increase the number of filters. The decoder path is made up 
    of multiple upsampling blocks, which increase the spatial 
    resolution of the feature maps and decrease the number of filters. 
    The decoder path also concatenates feature maps from the corresponding 
    encoder layer through skip connections, allowing the model to make use 
    of both low-level and high-level features.
    Reference: Ronneberger, O., Fischer, P., & Brox, T. (2015). 
    U-Net: Convolutional Networks for Biomedical Image Segmentation. 
    In International Conference on Medical image computing and 
    computer-assisted intervention (pp. 234-241). Springer, Cham.
    NOTE: This is based off an old version of Unet in 
    https://github.com/qubvel/segmentation_models.pytorch
    Parameters
    ----------
    backbone: string, default='resnet50'
        Name of classification model (without last dense layers) used 
        as feature extractor to build segmentation model. The backbone 
        is created using the timm wrapper (pre-training on ImageNet1K).
    encoder_freeze: boolean, default=False
        If ``True`` set all layers of encoder (backbone model) as non-trainable.
    pretrained: boolean, default=True
        If true, it download the weights of the pre-trained backbone network 
        in the ImageNet1K dataset.
    preprocessing: boolean, default=False
        If true, the preprocessing step is applied according to the mean 
        and standard deviation of the encoder model.
    non_trainable_layers: tuple, default=(0, 1, 2, 3, 4)
        Specifies which layers are non-trainable. 
        For example, if it is 3, the layer of index 3 in the given backbone network 
        are set as non-trainable. If the encoder_freeze parameter is set to True, 
        set this parameter.
    backbone_kwargs:
        pretrained (bool): load pretrained ImageNet-1k weights if true
        scriptable (bool): set layer config so that model is jit scriptable 
                           (not working for all models yet)
        exportable (bool): set layer config so that model is traceable 
                           ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit 
                       scripted layers (so far activations only)

        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific for more information;
        https://github.com/rwightman/pytorch-image-models
    backbone_indices: list, default=None
        Out indices, selects which indices to output.
    decoder_use_batchnorm: boolean, default=True
        If ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` 
        layers is used.    
    decoder_channels: tuple, default=(256, 128, 64, 32, 16)
        Tuple of numbers of ``Conv2D`` layer filters in decoder blocks.  
    in_channels: int, default=1
        Specifies the channel number of the image. By default it is 1. 
        If there is an RGB image input, please set it to 3.
    num_classes: int, default=5
        A number of classes for output (output shape - ``(batch, classes, h, w)``).
    center: boolean, default=False
        If ``True`` add ``Conv2dReLU`` block on encoder head.
    norm_layer: torch.nn.Module, default=nn.BatchNorm2d
        Is a layer that normalizes the activations of the previous layer.
    activation: torch.nn.Module, default=nn.ReLU
        An activation function to apply after the final convolution layer.
        Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, 
        **"tanh"**, **"identity"**, **relu**.
        Default is **relu**
    """
    def __init__(
            self,
            backbone='resnet50',
            encoder_freeze=False,
            pretrained=True,
            preprocessing=False,
            non_trainable_layers=(0, 1, 2, 3, 4),
            backbone_kwargs=None,
            backbone_indices=None,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=1,
            num_classes=5,
            center=False,
            norm_layer=nn.BatchNorm2d,
            activation=nn.ReLU
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        # NOTE some models need different backbone indices 
        # specified based on the alignment of features
        # and some models won't have a full enough range 
        # of feature strides to work properly.
        if backbone not in __available_models__:
            raise RuntimeError(
                f"Unknown backbone {backbone}!\n"
                f"Existing models should be selected " 
                f"from pretrained_backbones_unet import __available_models__"
            )

        encoder = create_model(
            backbone, features_only=True, out_indices=backbone_indices, 
            in_chans=in_channels, pretrained=pretrained, **backbone_kwargs
        )
        encoder_channels = [info["num_chs"] for info in encoder.feature_info][::-1]
        self.encoder = encoder
        if encoder_freeze: self._freeze_encoder(non_trainable_layers)
        if preprocessing:
            self.mean = self.encoder.default_cfg["mean"] 
            self.std = self.encoder.default_cfg["std"]
        else:
            self.mean = None
            self.std = None

        if not decoder_use_batchnorm:
            norm_layer = None
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=num_classes,
            norm_layer=norm_layer,
            center=center,
            activation=activation
        )

    def forward(self, x: torch.Tensor):
        if self.mean and self.std:
            x = self._preprocess_input(x)
        x = self.encoder(x)
        x.reverse()  
        x = self.decoder(x)
        return x

    @torch.no_grad()
    def predict(self, x):
        """
        Inference method. Switch model to `eval` mode, 
        call `.forward(x)` with `torch.no_grad()`
        Parameters
        ----------
        x: torch.Tensor
            4D torch tensor with shape (batch_size, channels, height, width)
        Returns
        ------- 
        prediction: torch.Tensor
            4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training: self.eval()
        x = self.forward(x)
        return x

    def _freeze_encoder(self, non_trainable_layer_idxs):
        """
        Set selected layers non trainable, excluding BatchNormalization layers.
        Parameters
        ----------
        non_trainable_layer_idxs: tuple
            Specifies which layers are non-trainable for 
            list of the non trainable layer names.
        """
        non_trainable_layers = [
            self.encoder.feature_info[layer_idx]["module"].replace(".", "_")\
             for layer_idx in non_trainable_layer_idxs
        ]
        for layer in non_trainable_layers:
            for child in self.encoder[layer].children():
                for param in child.parameters():
                    param.requires_grad = False
        return

    def _preprocess_input(self, x, input_range=[0, 1], inplace=False):
        if not x.is_floating_point():
            raise TypeError(f"Input tensor should be a float tensor. Got {x.dtype}.")

        if x.ndim < 3:
            raise ValueError(
                f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {x.size()}"
            )

        if not inplace:
            x = x.clone()

        dtype = x.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=x.device)
        if (std == 0).any():
            raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        return x.sub_(mean).div_(std)

# UNet Blocks

class Conv2dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, activation=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=False
        )
        self.bn = norm_layer(out_channels)
        self.act = activation(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, scale_factor=2.0, 
        activation=nn.ReLU, norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, activation=activation)
        self.scale_factor = scale_factor
        if norm_layer is None:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels,  **conv_args)
        else:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, norm_layer=norm_layer, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels, norm_layer=norm_layer, **conv_args)

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        if self.scale_factor != 1.0:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            norm_layer=nn.BatchNorm2d,
            center=True,
            activation=nn.ReLU
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = DecoderBlock(
                channels, channels, scale_factor=1.0, 
                activation=activation, norm_layer=norm_layer
            )
        else:
            self.center = nn.Identity()

        # list(decoder_channels[:-1][:len(encoder_channels)])
        in_channels = [in_chs + skip_chs for in_chs, skip_chs in zip(
            [encoder_channels[0]] + list(decoder_channels[:-1]),
            list(encoder_channels[1:]) + [0])]

        out_channels = decoder_channels

        if len(in_channels) != len(out_channels):
            in_channels.append(in_channels[-1]//2)

        self.blocks = nn.ModuleList()
        for in_chs, out_chs in zip(in_channels, out_channels):
            self.blocks.append(DecoderBlock(in_chs, out_chs, norm_layer=norm_layer))
        self.final_conv = nn.Conv2d(out_channels[-1], final_channels, kernel_size=(1, 1))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: List[torch.Tensor]):
        encoder_head = x[0]
        skips = x[1:]
        x = self.center(encoder_head)
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = b(x, skip)
        x = self.final_conv(x)
        return x