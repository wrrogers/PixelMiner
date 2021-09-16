import os
import re
import numpy as np
import matplotlib.pyplot as plt
from time import time


from ccc import concordance_correlation_coefficient
from scipy.stats import wilcoxon, ttest_rel, ttest_ind, mannwhitneyu, ranksums
from scipy.stats import f, shapiro, bartlett, f_oneway, kruskal
#from statsmodels.stats.weightstats import ztest

from glcm_mine import glcm_features
from radiomics import featureextractor, getTestCase
import SimpleITK as sitk

#from r_radiomics import get_features as get_features_r

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
#from skimage.measure import compare_ssim

import pandas as pd
from random import randint
from tqdm import tqdm
from PIL import Image, ImageOps, ImageEnhance

#from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import image

import matplotlib.pyplot as plt
from functools import partial
import multiprocessing

from get_binary_patch import get_binary_patch

import itertools

def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print(y.shape)
    f = np.var(x, ddof=1)/np.var(y, ddof=1) #calculate F test statistic 
    dfn = x.size-1 #define degrees of freedom numerator 
    dfd = y.size-1 #define degrees of freedom denominator 
    print(f)
    print(dfn)
    print(dfd)
    p = 1-f.cdf(f, dfn, dfd) #find p-value of F test statistic 
    return f, p

def get_features(img, mask):
    #(H, W) = img.shape
            
    #plt.imshow(img)
    #plt.show()
    
    #plt.imshow(mask)
    #plt.show()

    #print('Shapes:', img.shape, mask.shape, img.dtype, mask.dtype)
    #print('Minmax:', img.min(), img.max())
    #img = np.expand_dims(img, -1)
    #mask = np.expand_dims(mask, -1)
    sitk_img = sitk.GetImageFromArray(img)
    sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    params = 'settings.yaml'
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    features = {}
    #print('Getting features ...')
    #tick = time()
    features = extractor.execute(sitk_img, sitk_mask)
    #tock = time()
    #print('... done in', tock-tick, 'seconds.')
    #print(features)
    return features

def listfiles(path, verbose = False, absolute = True):
	filenames= os.listdir (path) # get all files' and folders' names in the current directory
	if verbose: print("Total files and folders:", len(filenames))
	result = []
	for filename in filenames: # loop through all the files and folders
		if not os.path.isdir(os.path.join(os.path.abspath(path), filename)): # check whether the current object is a folder or not
			result.append(filename)
	if absolute: result = [os.path.join(path, x) for x in result]
	return result

def unnormalize(img):
    img += 1
    img /= 2
    img *= (1024 + 3071)
    img -= 1024
    #img *= 255
    img = img.astype(np.uint8)
    print(img.min(), img.max())
    return img

def normalize(x):
    return (x-x.min())/(x.max()-x.min())


def mp_worker(inputs):
    #print('Starting a process ...')
    #tick = time()
    features = get_features(inputs[0], inputs[1])
    #print('Doing it ...', inputs[0].shape, inputs[1].shape)
    #tock = time()
    #print('Process took', tock - tick, 'seconds')
    return features

def get_all_features(path, target, mask):
    files  = listfiles(path, absolute=False)
    ids    = np.unique([file[:16] for file in files])
    
    all_features = []
    for n, id in enumerate(ids):
        
        print(id)
        
        id_files = [file for file in files if file[:16] == ids[n]]
        
        tru_file = [file for file in id_files if file.find(target) > -1][0]
        #msk_file = [file for file in id_files if file.find(mask) > -1][0]
        
        tru = np.load(os.path.join(path, tru_file)) 
        tru = unnormalize(tru)
        
        #msk = np.load(os.path.join(path, msk_file))
        msk = np.ones((512, 512))
        #msk[0, 0] = 0
        msk = msk.astype(np.uint32)
        
        features_tru = get_features(tru, msk)
        #features_tru = glcm_features(tru)
        
        features_tru = [float(item[1]) for key, item in zip(features_tru.keys(), features_tru.items()) if key[:11] != 'diagnostics']
        all_features.append(list(features_tru))
    
    return np.array(all_features)

def get_all_features_r(path, target, mask):
    files  = listfiles(path, absolute=False)
    ids    = np.unique([file[:16] for file in files])
    
    all_features = []
    for n, id in enumerate(ids):
        
        print(id)
        
        id_files = [file for file in files if file[:16] == ids[n]]
                
        tru_file = [file for file in id_files if file.find(target) > -1][0]
        msk_file = [file for file in id_files if file.find(mask) > -1][0]

        tru = np.load(os.path.join(path, tru_file)) 
        tru = unnormalize(tru)

        msk = np.load(os.path.join(path, msk_file)).astype(np.float32)

        coord = get_binary_patch(msk)
        img   = tru[coord[0]:coord[2], coord[1]:coord[3]]

        features_tru = get_features_r(img)
        features_tru = [float(item[1]) for key, item in zip(features_tru.keys(), features_tru.items()) if key[:11] != 'diagnostics']
        all_features.append(list(features_tru))
    
    return np.array(all_features)
    
def rmses(tru, itp):
    diff_itp = (itp - tru)
    rse_itp  = np.sqrt(diff_itp**2)
    rmse_itp = np.mean(rse_itp, axis=0)
    return rmse_itp

def wilcoxons(itp1, itp2):
    p_values = []
    for i in range(itp1.shape[1]):
        try:
            p_values.append(wilcoxon(itp1[:, i], itp2[:, i])[1])
        except:
            p_values.append(-1)
    return np.array(p_values)

def ttests(itp1, itp2):
    p_values = []
    for i in range(itp1.shape[1]):
        try:
            p_values.append(ttest_rel(itp1[:, i], itp2[:, i])[1])
        except:
            p_values.append(-1)
    return np.array(p_values)

def para_non_para(x, y, t):
    tests = np.empty(t.shape)
    for n, normal in enumerate(t):
        if normal:
            tests[n] = x[n]
        else:
            tests[n] = y[n]
    p = tests < 0.05
    return np.array(p)

def final(rmse1, rmse2, use):
    x = []
    for i in range(len(rmse1)):
        if use[i]:
            if rmse1[i] > rmse2[i]:
                x.append(1)
            else:
                x.append(-1)
        else:
            x.append(0)
    x = np.array(x)
    x = np.expand_dims(x, 0)
    return x

def get_rmses(tru, cnn, itp):
    rmse_cnn = rmses(tru, cnn)
    rmse_lin = rmses(tru, itp['Linear'])
    rmse_bsp = rmses(tru, itp['BSpline'])
    rmse_cos = rmses(tru, itp['Cosine'])
    rmse_nne = rmses(tru, itp['Nearest'])
    out = np.stack((rmse_cnn, rmse_lin, rmse_bsp, rmse_cos, rmse_nne))
    return out

def get_results(tru, cnn, itp):
    rmse_cnn = rmses(tru, cnn)
    rmse_lin = rmses(tru, itp['Linear'])
    rmse_bsp = rmses(tru, itp['BSpline'])
    rmse_cos = rmses(tru, itp['Cosine'])
    rmse_nne = rmses(tru, itp['Nearest'])
    
    p_lin = wilcoxons(cnn, itp['Linear'])
    p_bsp = wilcoxons(cnn, itp['BSpline'])
    p_cos = wilcoxons(cnn, itp['Cosine'])
    p_nne = wilcoxons(cnn, itp['Nearest'])
    
    t_lin = ttests(cnn, itp['Linear'])
    t_bsp = ttests(cnn, itp['BSpline'])
    t_cos = ttests(cnn, itp['Cosine'])
    t_nne = ttests(cnn, itp['Nearest'])
    
    #normal1 = [shapiro(tru[:, i])[0] for i in range(tru.shape[1])]
    normal2 = [shapiro(tru[:, i])[1] for i in range(tru.shape[1])]
    normal2 = np.array(normal2)
    normal = normal2 < .05
        
    tests_lin = para_non_para(t_lin, p_lin, normal)
    tests_bsp = para_non_para(t_bsp, p_bsp, normal)
    tests_cos = para_non_para(t_cos, p_cos, normal)
    tests_nne = para_non_para(t_nne, p_nne, normal)
    
    z_final_lin = final(rmse_lin, rmse_cnn, tests_lin)
    z_final_bsp = final(rmse_bsp, rmse_cnn, tests_bsp)
    z_final_cos = final(rmse_cos, rmse_cnn, tests_cos)
    z_final_nne = final(rmse_nne, rmse_cnn, tests_nne)
    
    z_final_stack = np.vstack((z_final_lin, z_final_bsp, z_final_nne, z_final_cos))
    return z_final_stack
    
def display_results(z_final_stack, features):
    fig, ax = plt.subplots(figsize=(36, 18), tight_layout=True)
    ax.imshow(z_final_stack.T, cmap='gray')
    end, start = ax.get_ylim()
    #print(start, end)
    ax.yaxis.set_ticks(np.arange(start+.5, end+.5))
    ax.tick_params(axis='y')
    ax.set_yticklabels(features[:, 1])
    ax.set_xticklabels(['Linear', 'BSpline', 'Nearest Neighbor', 'Window Sinc'])
    ax.xaxis.set_ticks(np.arange(0, 4))
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    plt.show()

def list_features(path  = r'H:\Data\W'):
    files = os.listdir(path)
    
    arr1   = np.load(os.path.join(path, files[14]))
    arr1   = unnormalize(arr1)
    arr2   = np.load(os.path.join(path, files[8]))
    
    features = get_features(arr1, arr2)
    features = [key for key in features.keys() if key.find('diagnostic') < 0]
    features = [feature[9:] for feature in features]
    features = [re.sub(r"(\w)([A-Z])", r"\1 \2", feature) for feature in features]
    features = [feature.split('_') for feature in features]
    features = np.array(features)
    return features



