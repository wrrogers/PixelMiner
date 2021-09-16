import os
import re
import numpy as np
import cv2
import pathlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from ccc import concordance_correlation_coefficient
from radiomics import featureextractor, getTestCase
import SimpleITK as sitk

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
#from skimage.measure import compare_ssim

import sys
import pandas as pd
import cv2
import concurrent
from random import randint
from tqdm import tqdm
from PIL import Image, ImageOps, ImageEnhance

#from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import image
from skimage.feature import greycomatrix, greycoprops

import pandas as pd

from functools import partial

import itertools

def load_scan(path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    read = reader.Execute()
    scan = sitk.GetArrayFromImage(read)
    return scan

def listfolders(path, verbose = False, absolute = True):
	filenames= os.listdir (path) # get all files' and folders' names in the current directory
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

def resample_image(img, out_spacing=(1.0, 1.0, 0.5), size=(512,512,7), base=12, n=5, interp=0):

    interps = [sitk.sitkBSpline, sitk.sitkLinear, sitk.sitkCosineWindowedSinc, sitk.sitkNearestNeighbor]

    size = img.shape
    itk_image = sitk.GetImageFromArray(img)
    #print('even image:', itk_image.GetSize())

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize((512,512,7))

    resample.SetInterpolator(interps[interp])
    resampled = resample.Execute(itk_image)

    #print('Resampled shape:', resampled.GetSize())

    return resampled

def interpolate_image(scan, base=12, position=5, interp=0):
    n = (scan.shape[0]//base) * position
    scan = scan[n-3:n+4]
    scan = scan[0::2, :, :]
    sample = resample_image(scan, out_spacing=(1,1,.5), interp=interp)
    imgs = sitk.GetArrayFromImage(sample)

    return imgs[3]

def get_slice(scan, base=12, position=5):
    n = (scan.shape[0]//base) * position
    return(scan[n])

def normalize(x):
    x = (x + x.min() + 1.)
    x = ((x-x.min())/(x.max()-x.min()))
    x = x * 2
    x = x - 1
    return x

def get_interpolations(img, base, position):
    print('Doing interpolations ...')

    methods = ['BSpline', 'Linear', 'NearestNeighbor', 'CosineWindowedSinc']

    iimgs = {}

    for n, method in enumerate(method):
        iimgs[method] = interpolate_image(img, base=base, position=position, interp=n)
    print('Done.\n')
    return iimgs

def unnormalize(img):
    img += 1
    img /= 2
    img *= (3071 + 1024)
    img -= 1024

    return img

def get_features(array, mask):
    (H, W) = array.shape
    #array = np.expand_dims(array, 0)
    #mask  = np.expand_dims(array, 0)
    print(array.shape, mask.shape)

    sitk_img = sitk.GetImageFromArray(array)
    #mask = np.ones((1, H, W), dtype=np.uint8)
    sitk_mask = sitk.GetImageFromArray(mask)
    params = 'settings.yaml'

    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    features = {}
    features = extractor.execute(sitk_img, sitk_mask)
    return features

###############################################################################

base_path = r'H:\Data\W'

div = 12
pos = 6

files = os.listdir(base_path)

files_tru = [file for file in files if file.find('tru') > -1]
files_tru = [file for file in files_tru if file.find('tru_one') < 0]

for i, tru_path in enumerate(files_tru):
    #if i > 0: break
    print(tru_path)
    id = tru_path[:16]
    print(id)

    tru = np.load(os.path.join(base_path, tru_path))
    print(tru.min(), tru.max())
    tru = unnormalize(tru)

    slice = tru[(tru.shape[0])//12*6]

    interpolations = get_interpolations(tru, div, pos)

    for key, interp in zip(interpolations.keys(), interpolations.items()):
        img = interp[1]
        print(key, img.shape)

        #features = get_features(img, msk)
        #features_int = [float(item[1]) for key, item in zip(features.keys(), features.items()) if key[:11] != 'diagnostics']

        img[img < -1024] = -1024
        img[img > 3071] = 3071
        img[0,0] = 3071
        img = normalize(img)

        np.save(os.path.join(base_path, id+'_'+key+'.npy'), img)

        print(os.path.join(base_path, id+'_'+key+'.npy'))

    print('----------------------------')
    #np.save(os.path.join(base_path, id+'_tru'+'.npy'), slice)





