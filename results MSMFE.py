import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from skimage import data
import pandas as pd
from itertools import product
from skimage.feature import greycomatrix, greycoprops
from scipy.stats import wilcoxon
from scipy.stats import binom_test

def normalize(x, scale=255):
    x = (((x-x.min())/(x.max()-x.min()))*scale)
    return x

class MSMFE:
    def __init__(self, ref, imgs=None, vmin=0, vmax=255, nbit=8, ks=5, verbose=False,
                features = ['Autocorrelation', 'ClusterProminence', 'ClusterShade', 'ClusterTendency', 'Contrast',
                            'Correlation', 'DifferenceEntropy', 'DifferenceVariance', 'Energy', 'Entropy',
                            'Id', 'Idm', 'Idmn', 'Idn', 'Imc1', 'Imc2', 'InverseVariance', 'JointAverage',
                            'MCC', 'MaximumProbability', 'SumAverage', 'SumEntropy', 'SumSquares']):

        if verbose: print('\tInitializing ...')
        ref = self.normalize(ref)
        if imgs is not None:
            self.keys = imgs.keys()
            for key in imgs.keys():
                imgs[key] = self.normalize(imgs[key])

        if verbose: print('\tCreating GLCM(s) ...')
        self.vmin = vmin
        self.vmax = vmax
        self.nbit = nbit
        self.ks   = ks
        self.glcm_ref = self.fast_glcm(ref)
        self.glcm_imgs = {}
        self.features = features
        self.error = {}
        self.img_feature_maps = {}
        self.feature_maps_ref = self.feature_maps(self.glcm_ref, features)
        self.imgs = imgs
        self.verbose=verbose

        if verbose: print('\tDone creating.')

    def get_names(self):
        names = list(self.keys) + ['_Reference']
        return names

    def normalize(self, img, max=1, scale=255):
        #Needs max set to one to account for PixelMiner not producing pixels up to 1
        img = (img - img.min())/(max-img.min())
        img *= scale
        #img = img.astype(np.uint8)
        return img

    def get_feature_maps(self):

        if self.imgs is not None:
            for key in self.keys:
                glcm = self.fast_glcm(self.imgs[key])
                self.img_feature_maps[key] = self.feature_maps(glcm, self.features)

            self.img_feature_maps['Reference'] = self.feature_maps_ref

            return self.img_feature_maps
        else:
            return self.feature_maps_ref

    def get_error(self, return_diff=False):
        if self.imgs is not None:
            for key in self.keys:
                glcm = self.fast_glcm(self.imgs[key])
                self.img_feature_maps[key] = self.feature_maps(glcm, self.features)
            if return_diff:
                diff_df  = pd.DataFrame(index=self.keys, columns=self.features)
            error_df = pd.DataFrame(index=self.keys, columns=self.features)
            for feature in self.features:
                #if self.verbose: print('\tDoing feature ...', feature, 'x'+str(len(self.keys)))
                for key in self.keys:
                    #print('\t\t'+key)
                    #print('\t\t'+str(self.img_feature_maps.keys()))
                    img = self.img_feature_maps[key][feature]
                    ref = self.feature_maps_ref[feature]
                    diff = ref - img
                    if return_diff:
                        diff_df.at[key, feature] = diff.mean()
                    error = ((diff) ** 2).mean()
                    error_df.at[key, feature] = error
            if return_diff:
                return error_df, diff_df
            else:
                return error_df
        else:
            print('Input needs an image and a reference image to calculate error.')

    def get_saliency(self, feature):
        saliencies = []
        for key in self.keys:
            img = self.feature_maps[feature][key]
            ref = self.feature_maps_ref[feature]
            saliencies.append((ref - img) ** 2)
        saliencies = np.asarray(saliencies)
        return saliencies

    def calculate_matrix(self, img, voxelCoordinates=None):
        r"""
        Compute GLCMs for the input image for every direction in 3D.
        Calculated GLCMs are placed in array P_glcm with shape (i/j, a)
        i/j = total gray-level bins for image array,
        a = directions in 3D (generated by imageoperations.generateAngles)
        """

        quant = normalize(img, scale=self.nbit).astype(np.int8)
        degrees  = [0, np.pi/4, np.pi/2, (3*np.pi)]
        distance = [1]
        P_glcm = greycomatrix(quant, distance, degrees, levels=self.nbit)
        P_glcm = np.moveaxis(P_glcm, -2, 0)
        P_glcm = P_glcm.astype(np.float32)

        sumP_glcm = np.sum(P_glcm, (1, 2)).astype(np.float32)

        sumP_glcm[sumP_glcm == 0] = np.nan
        P_glcm /= sumP_glcm[:, None, None, :]
        P_glcm = np.moveaxis(P_glcm, -1, 0).squeeze()
        return P_glcm

    def fast_glcm(self, img, conv=True, scale=False):

        min, max = self.vmin, self.vmax
        shape = img.shape
        if len(shape) > 2:
            print('Shape of', shape, 'is invalid, images must be 2d.')
            return
        h,w = img.shape

        # digitize
        bins = np.linspace(min, max, self.nbit+1)[1:]

        #print('Bins:', bins)

        gl   = np.digitize(img, bins) - 1
        gl.shape

        #print('Unique:', np.unique(gl))
        #print('GL:', gl.min(), gl.max())

        shifts = np.zeros((4, h, w))

        shifts[0] = np.append(       gl[:, 1:],        gl[:, -1:], axis=1) # one
        shifts[1] = np.append(       gl[1:, :],        gl[-1:, :], axis=0) # two
        shifts[2] = np.append(shifts[0][1:, :], shifts[0][-1:, :], axis=0) # three
        shifts[3] = np.append(shifts[0][:1, :], shifts[0][:-1, :], axis=0) # four

        #plt.imshow(gl)
        #plt.show()
        #plt.imshow(shifts[0])
        #plt.show()

        # make glcm
        glcm = np.zeros((4, self.nbit, self.nbit, h, w), dtype=np.uint8)
        for n, shift in enumerate(shifts):
            for i in range(self.nbit):
                for j in range(self.nbit):
                    mask = ((gl==i) & (shift==j))
                    glcm[n, i, j, mask] = 1

            if conv:
                kernel = np.ones((self.ks, self.ks), dtype=np.uint8)
                for i in range(self.nbit):
                    for j in range(self.nbit):
                        glcm[n, i, j] = cv2.filter2D(glcm[n, i, j], -1, kernel)

            glcm = glcm.astype(np.float32)

        if scale:
            matrix = self.calculate_matrix(img)
            #matrix = glcm.sum((3, 4))
            #print('SHAPE OF THE SCIKIT IMAGE MATRIX:', matrix.shape)
            glcm = matrix[:, :, :, None, None] * glcm

            #for direction in range(4):
            #    matrix[direction] = self.normalize(matrix[direction], scale=1)

        glcm = np.moveaxis(glcm, 0, -1)
        return glcm

    def get_means(self, img, glcm):
        h,w = img.shape

        mean_i = np.zeros((h,w), dtype=np.float32)
        for i in range(self.nbit):
            for j in range(self.nbit):
                mean_i += glcm[i,j] * i / (self.nbit)**2

        mean_j = np.zeros((h,w), dtype=np.float32)
        for j in range(self.nbit):
            for i in range(self.nbit):
                mean_j += glcm[i,j] * j / (self.nbit)**2

        return mean_i, mean_j

    def get_stds(self, img, glcm):
        h,w = img.shape

        mean_i, mean_j = self.get_means(img, glcm)

        std_i = np.zeros((h,w), dtype=np.float32)
        for i in range(self.nbit):
            for j in range(self.nbit):
                std_i += (glcm[i,j] * i - mean_i)**2
        std_i = np.sqrt(std_i)

        std_j = np.zeros((h,w), dtype=np.float32)
        for j in range(self.nbit):
            for i in range(self.nbit):
                std_j += (glcm[i,j] * j - mean_j)**2
        std_j = np.sqrt(std_j)

        return mean_i, mean_j, std_i, std_j

    def get_max(self, glcm):
            max_  = np.max(glcm, axis=(0,1))
            return(max_)

    def feature_maps(self, glcm, features):
        glcm = normalize(glcm, scale=2)
        #h, w = glcm.shape[-3], glcm.shape[-2]


        #glcm *= 16

        #print('GLCM:', glcm.min(), glcm.max())

        '''
        for q in range(4):
            count = 1
            for o in range(8):
                for p in range(8):
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplot(8, 8, count)
                    test = glcm[o, p, :, :, q]
                    plt.imshow(test, vmax=25)
                    count+=1
            plt.show()
        '''

        eps = np.spacing(1)

        bitVector = np.arange(0,self.nbit,1)
        i, j = np.meshgrid(bitVector, bitVector, indexing='ij', sparse=True)
        iAddj = i + j
        iSubj = np.abs(i-j)

        ux = i[:, :, None, None, None] * glcm
        uy = j[:, :, None, None, None] * glcm

        #print('UX, UY:', ux.shape, uy.shape, ux.min(), ux.max())

        '''
        for q in range(4):
            count = 1
            for o in range(8):
                for p in range(8):
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplot(8, 8, count)
                    test = ux[o, p, :, :, q]
                    plt.imshow(test, vmax=25)
                    count+=1
            plt.show()
        '''

        px = np.sum(glcm, 1)
        px = px[:, None, :, :, :]
        py = np.sum(glcm, 0)
        py = py[None, :, :, :, :]

        #for m in range(4):
        #    #plt.subplot(2,2,m+1)
        #    plt.title(str(ux[:, :, m].min()) + ' ' + str(ux [:, :, m].max()))
        #    plt.imshow(ux[:, :, m])
        #    plt.show()

        ux = np.sum((i[:, :, None, None, None] * glcm), (0, 1))
        ux = normalize(ux, scale=self.nbit)
        uy = np.sum((j[:, :, None, None, None] * glcm), (0, 1))
        uy = normalize(uy, scale=self.nbit)

        '''
        print()
        print('GLCM stuff:')
        print(glcm.min(), glcm.max())
        print()
        print('IJ Stuff:')
        print(i[:, :, None, None, None].shape)
        print(j[:, :, None, None, None].shape)
        print()
        print('U stuff:')
        print(ux.shape)
        print(uy.shape)

        for n in range(4):
            plt.title('ux')
            plt.imshow(ux[:, :, n])
            plt.show()
        '''

        kValuesSum  = np.arange(0, (self.nbit * 2)-1, dtype='float')
        #kValuesSum = np.arange(2, (self.nbit * 2) + 1, dtype='float')

        kDiagIntensity = np.array([iAddj == k for k in  kValuesSum])
        GLCMDiagIntensity = np.array([kDiagIntensity[int(k)][:, :, None, None, None] * glcm for k in kValuesSum])
        pxAddy = np.sum(GLCMDiagIntensity, (1, 2))

        kValuesDiff = np.arange(0, self.nbit, dtype='float')
        #kValuesDiff = np.arange(0, self.nbit, dtype='float')
        kDiagContrast = np.array([iSubj == k for k in  kValuesDiff])
        GLCMDiagIntensity = np.array([kDiagContrast[int(k)][:, :, None, None, None] * glcm for k in kValuesDiff])
        pxSuby = np.sum(GLCMDiagIntensity, (1, 2))

        HXY = (-1) * np.sum((glcm * np.log2(glcm + eps)), (0, 1))

        features_dict = {}

        if 'Autocorrelation' in features:
            ac = np.sum(glcm * (i * j)[:, :, None, None, None], (0, 1))
            features_dict['Autocorrelation'] = np.nanmean(ac, -1)

        if 'ClusterProminence' in features:
            cp = np.sum((glcm * (((i + j)[:, :, None, None, None] - ux - uy) ** 4)), (0, 1))
            features_dict['ClusterProminence'] = np.nanmean(cp, -1)

        if 'ClusterShade' in features:
            cs = np.sum((glcm * (((i + j)[:, :, None, None, None] - ux - uy) ** 3)), (0, 1))
            features_dict['ClusterShade'] = np.nanmean(cs, -1)

        if 'ClusterTendency' in features:
            ct = np.sum((glcm * (((i + j)[:, :, None, None, None] - ux - uy) ** 2)), (0, 1))
            features_dict['ClusterTendency'] = np.nanmean(ct, -1)

        if 'Contrast' in features:
            cont = np.sum((glcm * ((np.abs(i - j))[:, :, None, None, None] ** 2)), (0, 1))
            features_dict['Contrast'] = np.nanmean(cont, -1)

        if 'Correlation' in features:
            # shape = (Nv, 1, 1, angles)
            sigx = np.sum(glcm * ((i[:, :, None, None, None] - ux) ** 2), (0, 1), keepdims=True) ** 0.5
            # shape = (Nv, 1, 1, angles)
            sigy = np.sum(glcm * ((j[:, :, None, None, None] - uy) ** 2), (0, 1), keepdims=True) ** 0.5

            corm = np.sum(glcm * (i[:, :, None, None, None] - ux) * (j[:, :, None, None, None] - uy), (0, 1), keepdims=True)
            corr = corm / (sigx * sigy + eps)
            corr[sigx * sigy == 0] = 1  # Set elements that would be divided by 0 to 1.
            features_dict['Correlation'] = np.nanmean(corr, (0, 1, -1))

        if 'DifferenceAverage' in features:
            features_dict['DifferenceAverage'] = np.sum((kValuesDiff[:, None, None, None] * pxSuby), (0, -1))

        if 'DifferenceEntropy' in features:
            features_dict['DifferenceEntropy'] = (-1) * np.sum((pxSuby * np.log2(pxSuby + eps)), (0, -1))

        if 'DifferenceVariance' in features:
            diffavg = np.sum((kValuesDiff[:, None, None, None] * pxSuby), 0, keepdims=True)
            diffvar = np.sum((pxSuby * ((kValuesDiff[:, None, None, None] - diffavg) ** 2)), (0, -1))
            features_dict['DifferenceVariance'] = diffvar

        if 'Energy' in features:
            sum_squares = np.sum((glcm ** 2), (0, 1))
            features_dict['Energy'] = np.nanmean(sum_squares, -1)

        if 'Entropy' in features:
            features_dict['Entropy'] = np.sum(HXY, -1)

        if 'Id' in features:
            features_dict['Id'] = np.sum(pxSuby / (1 + kValuesDiff[:, None, None, None]), (0, -1))

        if 'Idm' in features:
            features_dict['Idm'] = np.sum(pxSuby / (1 + (kValuesDiff[:, None, None, None] ** 2)), (0, -1))

        if 'Idmn' in features:
            features_dict['Idmn'] = np.sum(pxSuby / (1 + ((kValuesDiff[:, None, None, None] ** 2) / (self.nbit ** 2))), (0,-1))

        if 'Idn' in features:
            features_dict['Idn'] = np.sum(pxSuby / (1 + (kValuesDiff[:, None, None, None] / self.nbit)), (0, -1))

        if 'Imc1' in features:
            # entropy of px # shape = (Nv, angles)
            HX = (-1) * np.sum((px * np.log2(px + eps)), (0, 1))
            # entropy of py # shape = (Nv, angles)
            HY = (-1) * np.sum((py * np.log2(py + eps)), (0, 1))
            # shape = (Nv, angles)
            HXY1 = (-1) * np.sum((glcm * np.log2(px * py + eps)), (0, 1))

            div = np.fmax(HX, HY)

            imc1 = HXY - HXY1
            imc1[div != 0] /= div[div != 0]
            imc1[div == 0] = 0  # Set elements that would be divided by 0 to 0

            features_dict['Imc1'] = np.nanmean(imc1, -1)

            #print('IMC1:', features_dict['Imc1'].shape)

        if 'Imc2' in features:
            # shape = (Nv, angles)
            HXY2 = (-1) * np.sum(((px * py) * np.log2(px * py + eps)), (0, 1))

            imc2 = (1 - np.e ** (-2 * (HXY2 - HXY)))

            min = imc2.min()
            imc2 += np.abs(min)
            #print(imc2.min(), imc2.max())

            imc2 = imc2 ** 0.5

            imc2[HXY2 == HXY] = 0

            features_dict['Imc2'] = np.nanmean(imc2, -1)

        if 'InverseVariance' in features:
            features_dict['InverseVariance'] = np.sum(pxSuby[1:, :, :, :] / kValuesDiff[1:, None, None, None] ** 2, (0, -1))  # Skip k = 0 (division by 0)

        if 'JointAverage' in features:
            features_dict['JointAverage'] = ux.mean(-1)

        if 'MCC' in features:
            # Calculate Q (shape (i, i, d)). To prevent division by 0, add epsilon (such a division can occur when in a ROI
            # along a certain angle, voxels with gray level i do not have neighbors
            nom = glcm[:, :, :, :, :] * glcm[:, :, :, :, :]
            den =   px[:, 0, :, :, :] *   py[:, 0, :, :, :]
            den = np.expand_dims(den, 1)

            Q = (nom /  (den + eps))  # sum over k (4th axis --> index 3)

            for gl in range(1, glcm.shape[1]):
                Q += ((glcm[:, None, gl, :, :] * glcm[None, :, gl, :, :]) /  # slice: v, i, j, k, d
                        (px[:, None,  0, :, :] *   py[None, :, gl, :, :] + eps))  # sum over k (4th axis --> index 3)

            #print('Q not Anon', Q.shape)

            # calculation of eigenvalues if performed on last 2 dimensions, therefore, move the angles dimension (d) forward
            Q_eigenValue = np.linalg.eigvals(Q.transpose((2, 3, 4, 0, 1)))
            Q_eigenValue.sort()  # sorts along last axis --> eigenvalues, low to high

            if Q_eigenValue.shape[3] < 2:
                return 1  # flat region

            #print(Q_eigenValue.shape)

            MCC = np.sqrt(Q_eigenValue[:, :, :,-2])  # 2nd highest eigenvalue

            #print(MCC.shape)

            features_dict['MCC'] = np.nanmean(MCC, 2).real

        if 'MaximumProbability' in features:
            maxprob = np.amax(glcm, (0, 1))
            features_dict['MaximumProbability'] = np.nanmean(maxprob, -1)

        if 'SumAverage' in features:
            sumavg = np.sum((kValuesSum[:, None, None, None] * pxAddy), 0)
            features_dict['SumAverage'] = np.nanmean(sumavg, -1)

        if 'SumEntropy' in features:
            sumentr = (-1) * np.sum((pxAddy * np.log2(pxAddy + eps)), 0)
            features_dict['SumEntropy'] = np.nanmean(sumentr, -1)

        if 'SumSquares' in features:
            ix = (i[:, :, None, None, None] - ux) ** 2
            ss = np.sum(glcm * ix, (0, 1))
            features_dict['SumSquares'] = np.nanmean(ss, -1)

        return features_dict

#if __name__ == '__main__':

def stringerate(number, length):
    number = str(number)
    strlen = len(number)
    zeros  = "0" * (length - strlen)
    return zeros + number

errors = []
for i in range(50):
    id = stringerate(i+1, 4)
    path = r'H:\Data\W\W_DATA_SET_W{}_tru_one.npy'.format(id)
    print(path)
    ref = np.load(path)

    imgs = {}
    imgs['PixelMiner'] = np.load(r'H:\Data\W\W_DATA_SET_W0001_PixelCNN.npy')
    imgs['Window Sinc'] = np.load(r'H:\Data\W\W_DATA_SET_W0001_CosineWindowedSinc.npy')
    imgs['Linear'] = np.load(r'H:\Data\W\W_DATA_SET_W0001_Linear.npy')
    imgs['BSpline'] = np.load(r'H:\Data\W\W_DATA_SET_W0001_BSpline.npy')
    imgs['Nearest Neighbors'] = np.load(r'H:\Data\W\W_DATA_SET_W0001_NearestNeighbor.npy')

    msmfe = MSMFE(ref, imgs, verbose=True)

    error = msmfe.get_error()
    error.to_csv(id+'.csv')
    e = error.to_numpy()
    errors.append(e)

    #feature_maps = msmfe.get_feature_maps()

    #feature_maps['contrast2'].min()
    #feature_maps['contrast2'].max()
    #plt.hist(feature_maps['contrast2'].flatten(), bins=256)

    '''
    plt.figure(figsize=(36,18))
    for i, key in enumerate(feature_maps.keys()):
        #print(i, key)
        #plt.subplot(4,6,i+1)
        img = feature_maps[key]
        plt.title(key + ' ' + str(img.min())+ " " +str(img.max()))
        #img = normalize(img, scale=256).astype(np.uint8)
        #print(img.shape, img.min(), img.max())
        plt.imshow(img[:, :])
        plt.show()
    '''

    len(errors)
    errors = np.array(errors)

    #np.save('MSMFE_errors:', errors)

    errors.shape

    n_errors = np.zeros((50, 5, 23))

    for i in range(23):
        n_errors[:, :, i] = normalize(errors[:, :, i], scale=1)

    n_errors[:, 0, :].mean()
    n_errors[:, 1, :].mean()
    n_errors[:, 2, :].mean()
    n_errors[:, 3, :].mean()
    n_errors[:, 4, :].mean()

    n_errors[:, 0, :].std()
    n_errors[:, 1, :].std()
    n_errors[:, 2, :].std()
    n_errors[:, 3, :].std()
    n_errors[:, 4, :].std()

    pm_ws = (n_errors[:, 0, :] < n_errors[:, 1, :]).sum() / (23 * 50)
    pm_ln = (n_errors[:, 0, :] < n_errors[:, 2, :]).sum() / (23 * 50)
    pm_bs = (n_errors[:, 0, :] < n_errors[:, 3, :]).sum() / (23 * 50)
    pm_nn = (n_errors[:, 0, :] < n_errors[:, 4, :]).sum() / (23 * 50)

    #pm_ws = (n_errors[:, 1, :] < n_errors[:, 0, :]).sum() / (23 * 50)
    nn_ln = (n_errors[:, 4, :] < n_errors[:, 2, :]).sum() / (23 * 50)
    nn_ws = (n_errors[:, 4, :] < n_errors[:, 1, :]).sum() / (23 * 50)
    nn_bs = (n_errors[:, 4, :] < n_errors[:, 3, :]).sum() / (23 * 50)

    bs_ln = (n_errors[:, 3, :] < n_errors[:, 2, :]).sum() / (23 * 50)
    bs_ws = (n_errors[:, 3, :] < n_errors[:, 1, :]).sum() / (23 * 50)
    bs_nn = (n_errors[:, 3, :] < n_errors[:, 4, :]).sum() / (23 * 50)

    ws_ln = (n_errors[:, 1, :] < n_errors[:, 2, :]).sum() / (23 * 50)
    ws_bs = (n_errors[:, 1, :] < n_errors[:, 3, :]).sum() / (23 * 50)
    ws_nn = (n_errors[:, 1, :] < n_errors[:, 4, :]).sum() / (23 * 50)

    ln_nn = (n_errors[:, 2, :] < n_errors[:, 4, :]).sum() / (23 * 50)

    print('\t-\t\t', 'PixelMiner \t Win Sinc \t BSpline \t Linear \t Nearest')
    print('PixleMiner\t\t',
          '\t-\t\t',
          round(pm_ws, 3), '\t\t',
          round(pm_bs, 3), '\t\t',
          round(pm_ln, 3), '\t\t',
          round(pm_nn, 3))

    print('Win Sinc\t\t',
          '\t-\t\t',
          '\t-\t\t',
          round(ws_bs, 3), '\t\t',
          round(ws_ln, 3), '\t\t',
          round(ws_nn, 3))

    print('BSpline\t\t\t',
          '\t-\t\t',
          '\t-\t\t',
          '\t-\t\t',
          round(bs_ln,3), '\t\t',
          round(bs_nn,3))

    print('Linear\t\t\t',
          '\t-\t\t',
          '\t-\t\t',
          '\t-\t\t',
          '\t-\t\t',
          round(ln_nn,3))

    print('Wilcoxons:')
    print('Win Sinc:', wilcoxon(n_errors[:, 0, :].flatten(), n_errors[:, 1, :].flatten()))
    print('Linear:', wilcoxon(n_errors[:, 0, :].flatten(), n_errors[:, 2, :].flatten()))
    print('BSpline:', wilcoxon(n_errors[:, 0, :].flatten(), n_errors[:, 3, :].flatten()))
    print('Nearest:', wilcoxon(n_errors[:, 0, :].flatten(), n_errors[:, 4, :].flatten()))

    shape = n_errors.shape[0] * n_errors.shape[2]

    print('Binomial test:')
    print('Win Sinc:', binom_test((n_errors[:, 0, :] > n_errors[:, 1, :]).sum() , shape))
    print('Linear:',   binom_test((n_errors[:, 0, :] > n_errors[:, 2, :]).sum() , shape))
    print('BSpline:',  binom_test((n_errors[:, 0, :] > n_errors[:, 3, :]).sum() , shape))
    print('Nearest:',  binom_test((n_errors[:, 0, :] > n_errors[:, 4, :]).sum() , shape))



