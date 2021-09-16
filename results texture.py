import os
import re
import numpy as np
import matplotlib.pyplot as plt

from ccc import concordance_correlation_coefficient
from scipy.stats import wilcoxon, ttest_rel, ttest_ind, mannwhitneyu, ranksums
from scipy.stats import f, shapiro, bartlett, f_oneway, kruskal
from statsmodels.stats.weightstats import ztest
from scipy.stats import binom_test

from radiomics import featureextractor, getTestCase
import SimpleITK as sitk

import pandas as pd
from random import randint
from tqdm import tqdm
from PIL import Image, ImageOps, ImageEnhance

#from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import image

import matplotlib.pyplot as plt
from functools import partial

from get_features_functions import get_features, final, get_results, display_results
from get_features_functions import listfiles, unnormalize, get_all_features
from get_features_functions import rmses, wilcoxons, ttests, para_non_para

from get_features_functions import get_all_features, get_rmses, normalize

def unnormalize(img):
    img += 1
    img /= 2
    img *= (1024 + 3071)
    #img -= 1024
    #img *= 255
    img = img.astype(np.uint8)
    print(img.min(), img.max())
    return img

def normalize(a):
    b = (a - np.min(a))/np.ptp(a)
    return b

def rse(x, y):
    diff = x - y
    sqrd = diff ** 2
    sqrt = np.sqrt(sqrd)
    return sqrt

n = 0
path  = r'H:\Data\W'
files = os.listdir(path)
print(files[14])
arr1   = np.load(os.path.join(path, files[14]))
arr1   = unnormalize(arr1)
print(arr1.dtype, arr1.min(), arr1.max())
print(files[8])
arr2   = np.load(os.path.join(path, files[8]))

features = get_features(arr1, arr2)
features = [key for key in features.keys() if key.find('diagnostic') < 0]
features = [feature[9:] for feature in features]
features = [re.sub(r"(\w)([A-Z])", r"\1 \2", feature) for feature in features]
features = [feature.split('_') for feature in features]
features = np.array(features)

lung_itp = {}

lung_tru               = get_all_features(path, 'tru_one', 'lung')
lung_cnn               = get_all_features(path, 'PixelCNN', 'lung')
lung_itp['Linear']     = get_all_features(path, 'Linear', 'lung')  
lung_itp['BSpline']    = get_all_features(path, 'BSpline', 'lung')  
lung_itp['Cosine']     = get_all_features(path, 'Cosine', 'lung')  
lung_itp['Nearest']    = get_all_features(path, 'Nearest', 'lung')
    
lung_results = get_results(lung_tru, lung_cnn, lung_itp)
display_results(lung_results, features)

cnn_diff = rse(lung_tru, lung_cnn)
lin_diff = rse(lung_tru, lung_itp['Linear'])
cos_diff = rse(lung_tru, lung_itp['Cosine'])
ner_diff = rse(lung_tru, lung_itp['Nearest'])
bsp_diff = rse(lung_tru, lung_itp['BSpline'])

t = cnn_diff.shape[0] * cnn_diff.shape[1]

print()
print('Percent Greater:')
print('Linear\t\t\t Win Sinc\t\t\t Nearest\t\t\t BSpline\t\t\t PixelMiner')
print('\t-\t\t\t'                    , (lin_diff < cos_diff).sum() / t, (lin_diff < ner_diff).sum() / t, (lin_diff < bsp_diff).sum() / t, (lin_diff < cnn_diff).sum() / t)
print((cos_diff < lin_diff).sum() / t, '\t-\t\t\t'                    , (cos_diff < ner_diff).sum() / t, (cos_diff < bsp_diff).sum() / t, (cos_diff < cnn_diff).sum() / t)
print((ner_diff < lin_diff).sum() / t, (ner_diff < cos_diff).sum() / t, '\t-\t\t\t'                    , (ner_diff < bsp_diff).sum() / t, (ner_diff < cnn_diff).sum() / t)
print((bsp_diff < lin_diff).sum() / t, (bsp_diff < cos_diff).sum() / t, (bsp_diff < ner_diff).sum() / t, '\t-\t\t\t'                    , (bsp_diff < cnn_diff).sum() / t)
print((cnn_diff < lin_diff).sum() / t, (cnn_diff < cos_diff).sum() / t, (cnn_diff < ner_diff).sum() / t, (cnn_diff < bsp_diff).sum() / t, '\t-\t\t'                      )

error = np.array([cnn_diff, lin_diff, cos_diff, ner_diff, bsp_diff])
n_error = np.zeros((5, 50, 51))
for i in range(error.shape[-1]):
    n_error[:, :, i] = normalize(error[:, :, i])

print()
print('NRMSE Mean:')
print('PixelMiner:', n_error[0].mean())
print('Linear:', n_error[1].mean())
print('Win Sinc:', n_error[2].mean())
print('Nearest:', n_error[3].mean())
print('BSpline', n_error[4].mean())

print()
print('NRMSE STD:')
print('PixelMiner:', n_error[0].std())
print('Linear:', n_error[1].std())
print('Win Sinc:', n_error[2].std())
print('Nearest:', n_error[3].std())
print('BSpline', n_error[4].std())


ccc_cnn = np.array([concordance_correlation_coefficient(lung_tru[:, i], lung_cnn[:, i]) for i in range(lung_tru.shape[1])])
ccc_lin = np.array([concordance_correlation_coefficient(lung_tru[:, i], lung_itp['Linear'][:, i]) for i in range(lung_tru.shape[1])])
ccc_bsp = np.array([concordance_correlation_coefficient(lung_tru[:, i], lung_itp['BSpline'][:, i]) for i in range(lung_tru.shape[1])])
ccc_ws  = np.array([concordance_correlation_coefficient(lung_tru[:, i], lung_itp['Cosine'][:, i]) for i in range(lung_tru.shape[1])])
ccc_nn  = np.array([concordance_correlation_coefficient(lung_tru[:, i], lung_itp['Nearest'][:, i]) for i in range(lung_tru.shape[1])])

cccs = np.vstack((ccc_cnn, ccc_bsp, ccc_nn, ccc_ws, ccc_lin))

print('Mean CCC')
print('PixelMiner', cccs[0].mean(), '\n'
      'Win Sinc', cccs[3].mean(),  '\n'
      'BSpline', cccs[1].mean(),  '\n'
      'Nearest', cccs[2].mean(),  '\n'
      'Linear', cccs[4].mean())

print()
print('Reproducibility')
thresh = .85
print('PixelMiner', (cccs[0] > thresh).sum() / 51, '\n'
      'Win Sinc', (cccs[3] > thresh).sum() / 51,  '\n'
      'BSpline', (cccs[1] > thresh).sum() / 51,  '\n'
      'Nearest', (cccs[2] > thresh).sum() / 51,  '\n'
      'Linear', (cccs[4] > thresh).sum() / 51)
      

print('Wilcoxons:')
print('Win Sinc:', wilcoxon(n_error[:, 0, :].flatten(), n_error[:, 2, :].flatten()))
print('Linear:', wilcoxon(n_error[:, 0, :].flatten(), n_error[:, 1, :].flatten()))
print('BSpline:', wilcoxon(n_error[:, 0, :].flatten(), n_error[:, 4, :].flatten()))
print('Nearest:', wilcoxon(n_error[:, 0, :].flatten(), n_error[:, 3, :].flatten()))

shape = n_error.shape[0] * n_error.shape[2]

print('Binomial test:')
print('Win Sinc:', binom_test((n_error[:, 0, :] < n_error[:, 2, :]).sum() , shape))
print('Linear:',   binom_test((n_error[:, 0, :] < n_error[:, 1, :]).sum() , shape))
print('BSpline:',  binom_test((n_error[:, 0, :] < n_error[:, 4, :]).sum() , shape))
print('Nearest:',  binom_test((n_error[:, 0, :] < n_error[:, 3, :]).sum() , shape))














