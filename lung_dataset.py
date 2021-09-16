import os
import sys
import numpy as np
import cv2
import concurrent
from random import randint
from tqdm import tqdm
from PIL import Image,ImageOps,ImageEnhance

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import image

from torch.nn import Parameter
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset,DataLoader,Subset

import matplotlib.pyplot as plt

from functools import partial

from parameters import Parameters
args = Parameters()

def preprocess(x, n_bits=False):
    x = x.float()               # 1 convert data to float
    if n_bits:
        x = x.div(2**n_bits - 1)    # 2 normalize to [0,1] given quantization
    x = x.mul(2).add(-1)        # 3 (should be between -1 and 1)
    return x

def deprocess(x, n_bits):
    x = x.add(1).div(2)         # 1. shift to [0,1]
    x = x.mul((2**n_bits) - 1)    # 2. quantize to n_bits
    x = x.long()                # 3. convert data to long
    return x

def listfolders(path, verbose = True, absolute = True):
    if verbose: print("Getting images from ...", path)
    filenames= os.listdir(path) # get all files' and folders' names in the current directory
    if verbose: print("Total files and folders:", len(filenames))
    result = []
    for filename in filenames: # loop through all the files and folders
        if os.path.isdir(os.path.join(os.path.abspath(path), filename)): # check whether the current object is a folder or not
            result.append(filename)
    if verbose: print("Total folders:", len(result))
    if verbose: print("Total files:", len(filenames) - len(result), '\n')
    if absolute: result = [os.path.join(path, x) for x in result]
    return result

def file_sort(files):
    nums = np.array([int(file[:file.find('.')]) for file in files])
    new_range = list(range(nums.min(), nums.max()+1))
    new_files = [str(n)+'.bmp' for n in new_range]
    return new_files

#img_path = r'C:\Users\w.rogers\Desktop\Data\Lung1'
#folders = listfolders(img_path)
#folder = folders[164]
#files = os.listdir(folder)
#files = [file for file in files if file[-3:] == 'bmp']
#new_files = file_sort(files)
#imgs = get_images()
#print(len(imgs), imgs[0].min(), imgs[0].max())
#plt.imshow(imgs[0][10])

class LungDataset(Dataset):
    def __init__(self, path=r'F:\William\Pulmonary Embolism Detection\train_png',
                 init_transform=None, transform=None, split=False, train=True,
                 spacing=0, max=None):
        #print("\n\nCreate a training set:", train)
        self.spacing = spacing
        folders = os.listdir(path)
        folders = [os.path.join(path, folder) for folder in folders]

        self.spacing = spacing

        #ids = np.arange(0, len(self.imgs), 1)

        if split:
            X_train, X_test = train_test_split(folders, test_size=.02, train_size=.98, shuffle=False)
            if train:
                self.paths = X_train
            else:
                #print('using test data')
                self.paths = X_test

        self.batch_transform = transform

        if max:
            self.paths = np.array(self.paths)
            self.paths = self.paths[:max]

        print("\n... LUNG1 Dataset Intialized with", len(self.paths), "scans")


    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        path =  self.paths[idx]
        files = os.listdir(path)

        n_files = len(files)
        n = randint(1+self.spacing, n_files-2-self.spacing)
        #n = 6

        try:
            file_one = files[n-1-self.spacing]
            file_two = files[n]
            file_tre = files[n+1+self.spacing]
        except:
            print('n files:', len(files))
            print('min max:', n-1-self.spacing, n+1+self.spacing)

        one = Image.open(os.path.join(path, file_one)).convert('L')
        two = Image.open(os.path.join(path, file_two)).convert('L')
        tre = Image.open(os.path.join(path, file_tre)).convert('L')

        imgs = Image.merge("RGB", (one, two, tre))

        if self.batch_transform:
            #print("doing transform ...")
            imgs = self.batch_transform(imgs)

        return imgs

def load_images(batch_size=32, split=False, train=True):
    while True:
        for ii, data in enumerate(create_data_loader(batch_size, split=split, train=train)):
            yield data

def create_data_loader(batch_size, split=False, train=True):
    transform = transforms.Compose([ #transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomCrop((64,64), padding_mode='edge'),
                                     transforms.ToTensor(),
                                     #transforms.Normalize(mean=[0, 0, 0],std=[1, 1, 1]),
                                     #lambda x: x.mul(255).div(2**(8-args.n_bits)).floor(),
                                     partial(preprocess)    # to model space [-1,1]
                                    ])

    train_set = LungDataset(split=split, transform=transform, train=train)

    train_loader = DataLoader(train_set, shuffle=False, batch_size=batch_size,
                              num_workers=0, pin_memory=True)

    #print("The length of the training set is", len(train_set))
    return train_loader

# --------------------
# Main
# --------------------

if __name__ == '__main__':
    training = create_data_loader(batch_size=32, split=True, train=True)
    data = next(iter(training))
    print(data.shape, data.min(), data.max())
    plt.figure(figsize=(16, 16))
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(data[0][i])
    plt.show()




