import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from scipy.stats import binom_test, wilcoxon

def stringerate(number, length):
    number = str(number)
    strlen = len(number)
    zeros  = "0" * (length - strlen)
    return zeros + number

def epr(ref, img):
    img1 = cv2.Canny(ref, 100, 200)
    img2 = cv2.Canny(img, 100, 200)
    img1[img1 < 255]  = 0
    img1[img1 >= 255] = 1
    img2[img2 < 255] = 0
    img2[img2 >= 255] = 1   
    nom = img1 + img2
    nom = nom.sum()
    den = img2.sum()
    return den/nom

path = r'H:\Data\Combined'

files = os.listdir(path)

ids = [file[:16] for file in files]
ids = np.unique(ids)
results = []
for n, id in enumerate(ids):
    #if n > 0: break
    n+=1
    r'H:\Data\W\W_DATA_SET_W'+stringerate(n, 4)+'_tru_one.npy'    

    tru  = np.load(os.path.join(path, id + '_tru_one.npy'))
    cnn  = np.load(os.path.join(path, id + '_PixelCNN.npy'))
    lin  = np.load(os.path.join(path, id + '_Linear.npy'))
    wis  = np.load(os.path.join(path, id + '_CosineWindowedSinc.npy'))
    ner  = np.load(os.path.join(path, id + '_NearestNeighbor.npy'))
    bsp  = np.load(os.path.join(path, id + '_BSpline.npy'))

    tru  = (((tru + 1) / 2) * 255).astype(np.uint8)
    cnn  = (((cnn + 1) / 2) * 255).astype(np.uint8)
    lin  = (((lin + 1) / 2) * 255).astype(np.uint8)
    wis  = (((wis + 1) / 2) * 255).astype(np.uint8)
    ner  = (((ner + 1) / 2) * 255).astype(np.uint8)
    bsp  = (((bsp + 1) / 2) * 255).astype(np.uint8)
    
    cnn_epr = epr(tru, cnn)
    lin_epr = epr(tru, lin)
    wis_epr = epr(tru, wis)
    ner_epr = epr(tru, ner)
    bsp_epr = epr(tru, bsp)
    
    results.append([cnn_epr, lin_epr, wis_epr, bsp_epr, ner_epr])

results = np.array(results)

cnn_diff, lin_diff, cos_diff, bsp_diff, ner_diff = results[:, 0], results[:, 1], results[:, 2], results[:, 3], results[:, 4]  
total = results.shape[0]

print((lin_diff > cos_diff).sum() / total, (lin_diff > bsp_diff).sum() / total, (lin_diff > cos_diff).sum() / total, (lin_diff > cnn_diff).sum() / total)
print((ner_diff > lin_diff).sum() / total, (ner_diff > bsp_diff).sum() / total, (ner_diff > cos_diff).sum() / total, (ner_diff > cnn_diff).sum() / total)
print((bsp_diff > lin_diff).sum() / total, (bsp_diff > ner_diff).sum() / total, (bsp_diff > cos_diff).sum() / total, (bsp_diff > cnn_diff).sum() / total)
print((cos_diff > lin_diff).sum() / total, (cos_diff > bsp_diff).sum() / total, (cos_diff > bsp_diff).sum() / total, (cos_diff > cnn_diff).sum() / total)
print((cnn_diff > lin_diff).sum() / total, (cnn_diff > bsp_diff).sum() / total, (cnn_diff > cos_diff).sum() / total, (cnn_diff > ner_diff).sum() / total)

(cos_diff >= cnn_diff).sum()/total
(cnn_diff >= cos_diff).sum()/total

(bsp_diff >= cnn_diff).sum()/total
(cnn_diff >= bsp_diff).sum()/total

(ner_diff >= cnn_diff).sum()/total
(cnn_diff >= ner_diff).sum()/total

(cos_diff >= bsp_diff).sum()/total
(bsp_diff >= cos_diff).sum()/total

(cos_diff >= ner_diff).sum()/total
(ner_diff >= cos_diff).sum()/total

(bsp_diff >= ner_diff).sum()/total
(ner_diff >= bsp_diff).sum()/total


###################################################################################################

print('\nResults:')
mean_results = results.mean(axis=0)
print()
print('PM:', mean_results[0], '\n',
      'NN:', mean_results[1],  '\n',
      'LN:', mean_results[2],  '\n',
      'WS:', mean_results[3],  '\n',
      'BS:', mean_results[4])

print('\nSandard Deviations and Frequencies:')
print('PM:', np.std(results[:, 0]), '\n',
      'NN:', np.std(results[:,1]), (1- (results[:,0] > results[:,1]).sum() / 50),  '\n',
      'LN:', np.std(results[:,2]), (1- (results[:,0] > results[:,2]).sum() / 50),  '\n',
      'WS:', np.std(results[:,3]), (1- (results[:,0] > results[:,3]).sum() / 50),  '\n',
      'BS:', np.std(results[:,4]), (1- (results[:,0] > results[:,4]).sum() / 50))

print('\nWilcoxons:')
print('NN:', wilcoxon(results[:,0], results[:,1]), '\n',
      'LN:', wilcoxon(results[:,0], results[:,2]), '\n',
      'WS:', wilcoxon(results[:,0], results[:,3]), '\n',
      'BS:', wilcoxon(results[:,0], results[:,4]))

bin_lin = (results[:,0] > results[:,1]).sum()
bin_wis = (results[:,0] > results[:,2]).sum()
bin_bsp = (results[:,0] > results[:,3]).sum()
bin_ner = (results[:,0] > results[:,4]).sum()

print('\nBinomials:')
print('LN:', binom_test(bin_lin, results.shape[0]), '\n',
      'WS:', binom_test(bin_wis, results.shape[0]), '\n',
      'BS:', binom_test(bin_bsp, results.shape[0]), '\n',
      'NN:', binom_test(bin_ner, results.shape[0]))




