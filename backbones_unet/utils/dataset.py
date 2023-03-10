import torch
from torch.utils.data import Dataset
import numpy as np
import os
import re
from PIL import Image


class SemanticSegmentationDataset(Dataset):
    def __init__(self, 
                 img_paths, 
                 mask_paths=None,
                 size=(256, 256),
                 mode='binary',
                 normalize=None):
        """
        Example semantic segmentation Dataset class.
        Run once when instantiating the Dataset object.
        If you want to use it for binary semantic segmentation, 
        please select the mode as 'binary'. For multi-class, enter 'multi'.
        example_data/
            └── /images/
                    └── 0001.png
                    └── 0002.png
                    └── 0003.png
                    └── ...
                /masks/
                    └── 0001_mask.png
                    └── 0002_mask.png
                    └── 0003_mask.png
                    └── ...
        img_paths : str
            The file path indicating the main directory that contains only images.
        mask_paths : str, default=None
            The file path indicating the main directory that contains only 
            ground truth images.
        size : tuple, default=(256, 256)
            Enter the (width, height) values into a tuple for resizing the data.
        mode : str, default='binary'
            Choose how the DataSet object should generate data. 
            Enter 'binary' for binary masks.
        normalize : orchvision.transforms.Normalize, default=None
            Normalize a tensor image with mean and standard deviation. 
            This transform does not support PIL Image.
        """
        self.img_paths = self._get_file_dir(img_paths)
        self.mask_paths = self._get_file_dir(mask_paths) if mask_paths is not None else mask_paths
        self.size = size
        self.mode = mode
        self.normalize = normalize
        
    def __len__(self):
        """
        Returns the number of samples in our dataset.
        Returns
        -------    
        num_datas : int    
            Number of datas.
        """
        return len(self.img_paths)
    
    def __getitem__(self, index):
        """
        Loads and returns a sample from the dataset at 
        the given index idx. Based on the index, it 
        identifies the image’s location on disk, 
        converts that to a tensor using read_image, 
        retrieves the corresponding label from the 
        ground truth data in self.mask_paths, calls the transform 
        functions on them (if applicable), and returns 
        the tensor image and corresponding label in a tuple.
        Returns
        -------   
        img, mask : torch.Tensor
            The transformed image and its corresponding 
            mask image. If the mask path is None, it 
            will only return the transformed image.
            output_shape_mask: (batch_size, 1, img_size, img_size)
            output_shape_img: (batch_size, 3, img_size, img_size)
        """
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.size[0], self.size[1])) 
        img = torch.Tensor(np.array(img, dtype=np.uint8).transpose((2, 0, 1)))
        if self.mask_paths is not None:
            mask_path = self.mask_paths[index]
            mask = Image.open(mask_path)
            mask = mask.resize((self.size[0], self.size[1])) 
            mask = np.array(mask)
            
            if self.mode == 'binary':
                mask = self._binary_mask(mask)
            else: 
                mask = self._multi_class_mask(mask)

            mask = torch.as_tensor(mask, dtype=torch.uint8)
            if self.normalize: img = self.normalize(img)
            return img, mask
        else:
            if self.normalize: img = self.normalize(img)
            return img

    def _multi_class_mask(self, mask):
        obj_ids = np.unique(mask)[1:]
        masks = mask == obj_ids[:, None, None]
        return masks

    def _binary_mask(self, mask):
        mask[:, :][mask[:, :] >= 1] = 1
        mask[:, :][mask[:, :] < 1] = 0
        mask = np.expand_dims(mask, axis=0)
        return mask

    def _get_file_dir(self, directory):
        """
        Returns files in the entered directory.
        Parameters
        ----------
        directory : string
            File path.
        Returns
        -------
        directories: list
            All files in the directory.
        """
        def atoi(text):
            return int(text) if text.isdigit() else text
            
        def natural_keys(text):
            return [atoi(c) for c in re.split('(\d+)',text)]

        for roots,dirs,files in os.walk(directory):               
            if files:
                directories = [roots + os.sep + file for file in  files]
                directories.sort(key=natural_keys)

        return directories