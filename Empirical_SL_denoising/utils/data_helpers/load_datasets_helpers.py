import os
import os.path
import numpy as np
import h5py
import torch
import torchvision.transforms as transforms
import PIL.Image as Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super().__init__()
        self.h5f = h5py.File(filename, "r")
        self.keys = list(self.h5f.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        data = np.array(self.h5f[key])
        return torch.Tensor(data)
    
class ImagenetSubdataset(torch.utils.data.Dataset):
    def __init__(self, size, path_to_ImageNet_train, mode='train'):
        super().__init__()
        load_path = './training_set_lists/'
        self.path_to_ImageNet_train = path_to_ImageNet_train
        if mode=='train':
            self.files = torch.load(load_path+f'trsize{size}_filepaths.pt') 
            self.transform = transforms.Compose([      
                transforms.CenterCrop(256),
                transforms.ToTensor(),                  
                ]) 
        elif mode=='val':
            self.files = torch.load(load_path+f'ImageNetVal{size}_filepaths.pt') 
            self.transform = transforms.Compose([      
                transforms.ToTensor(),                  
                ]) 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.path_to_ImageNet_train + self.files[index]).convert("RGB")
        data = self.transform(image)
        return data

