import gc
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import nvgpu

import torch
from torchvision import transforms
from torch.optim import Adam

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.measure import compare_ssim

#from get_windows import *
from get_windows import get_windows, get_side, get_bottom

from pixelcnnpp3d_final_v3 import sample_from_discretized_mix_logistic_1d, PixelCNNpp

from phantom_dataset import create_data_loader, get_ids

from parameters import Parameters

import SimpleITK as sitk

args = Parameters()

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = structural_similarity(imageA, imageB)
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
	# show the images
	plt.show()

def normalize(x):
    x = (x + x.min() + 1.)
    x = ((x-x.min())/(x.max()-x.min()))
    x = x * 2
    x = x - 1
    return x

def normalize256(x):
    x = (x + x.min() + 1.)
    x = ((x-x.min())/(x.max()-x.min()))
    x = x * 255
    x = x.astype(np.uint8)
    return x

def get_patch(img, window=(0, 0), patch_size=(64, 64)):
    half_patch = (patch_size[0]//2, patch_size[1]//2)
    #nx_patches = (img.shape[0]//patch_size[0]) + ((img.shape[0]-half_patch[0])//patch_size[0])
    #ny_patches = (img.shape[1]//patch_size[1]) + ((img.shape[1]-half_patch[1])//patch_size[1])
    
    patch = img[half_patch[0]*window[0]:half_patch[0]*window[0]+64, 
                half_patch[1]*window[1]:half_patch[1]*window[1]+64]
    patch = patch.astype(np.float32)
    patch = torch.from_numpy(patch)
    return patch

def get_corners(img, n=0, patch_size=(64, 64)):
    bottom_patch = img[-patch_size[0]:, :patch_size[1]]
    side_patch   = img[:patch_size[1], -patch_size[0]:]
    return bottom_patch, side_patch

def update_whole(img, patch, window=(0, 0), patch_size=(64, 64)):
    half_patch = (patch_size[0]//2, patch_size[1]//2)
    img[half_patch[0]*window[0]:half_patch[0]*window[0]+64, 
        half_patch[1]*window[1]:half_patch[1]*window[1]+64] = patch.detach().cpu().numpy()
    return img
    
'''
patch = get_patch(whole_img, window=(1, 0))
plt.imshow(patch)

patch = get_patch(whole_img, window=(0, 1))
plt.imshow(patch)
'''

def show_series(imgs):
    #imgs[imgs> 1] = 1
    #imgs[imgs<-1] = -1
    shape = tuple([imgs.shape[-1], imgs.shape[-1]*imgs.shape[1]])
    board = np.zeros(shape)
    print(board.shape)
    plt.figure(figsize=(12,12))
    for i in range(imgs.shape[1]):
        board[:, imgs.shape[-1]*i:imgs.shape[-1]*i+imgs.shape[-1]] = imgs[0,i]
    plt.imshow(board)
    plt.show()

def generate3d(targ, ymin=0, xmin=0, image_dims=(3,64,64), h=None, zero=True, 
               convert=False, display=False, erase=True):
    if convert:
        targ = torch.from_numpy(targ)        
    output = targ.clone().detach().cuda().float()
    if erase:
        output[:,:,1,ymin:,:] = 0
    with tqdm(total=(image_dims[1])*(image_dims[2]), desc='Generating {} image(s)'.format(targ.size(0))) as pbar:
        for yi in range(image_dims[1]):
            if yi < ymin:
                continue

            for xi in range(image_dims[2]):
                if xi < xmin:
                    continue
                logits = model(output, None)
                #print('\n\nLOGITS:', logits.size())
                sample = sample_from_discretized_mix_logistic_1d(logits)[:,:,yi,xi]                
                #print('\n\nSAMPLE:', sample.size())
                pixel = sample.clone().detach().requires_grad_(True)
                #print('\n\nOUTPUT:', output.size())
                #print('store:', output[:,:,1,yi,xi].size())
                #print('Pixel:', pixel.size())
                #print('INPUT:', pixel.unsqueeze(-1).size())
                output[:,:,1,yi,xi] = pixel                 
                if display:
                    if xi == 63:
                        plt.figure(figsize=(18, 18))
                        for i in range(3):
                            plt.subplot(1, 3, i+1)
                            temp = np.moveaxis(output[:, :, 1].detach().cpu().numpy().squeeze(), 0, -1).T
                            plt.imshow(temp, cmap='gray')
                        plt.show()
                pbar.update()                
    return output

def load_scan(path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    scan = sitk.GetArrayFromImage(image)
    return scan

image_dims = (3, 64, 64)
h = None

#path = r'D:\Data\MILDBL_ANON\MILD_ - 4280L090742\Thorax Screening (Adult)\LungCARE 1.0 B50f - 4'
#path = r'H:\Data\Stanford-CT-Lung\NSCLC Radiogenomics\AMC-001\1.3.6.1.4.1.14519.5.2.1.4334.1501.227933499470131058806289574760\1.3.6.1.4.1.14519.5.2.1.4334.1501.131836349235351218393791897864'
#path = r'H:\Data\Stanford-CT-Lung\NSCLC Radiogenomics\AMC-002\1.3.6.1.4.1.14519.5.2.1.4334.1501.318107938099351952423306401630\1.3.6.1.4.1.14519.5.2.1.4334.1501.426581693091941145906372008618'
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-422\07-25-2010-06557\0.000000-42364' #[66, 96:-128, :]
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-421\07-17-2010-30075\0.000000-00327'
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-420\07-31-2010-42181\0.000000-88538' #[100,160:-128,64:-64]
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-419\07-17-2010-87475\0.000000-26917' #[98,160:-160,64:-64]
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-418\07-15-2010-60673\0.000000-89047' #[98,160:-160,64:-64]
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-417\07-22-2010-22742\0.000000-27827' #[98,160:-160,64:-64]
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-416\07-30-2010-71000\0.000000-44349' #[98,192:-160,64:-64]
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-415\07-04-2010-18546\0.000000-77806' #[86,160:-160,96:-64]
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-414\07-10-2010-66211\0.000000-70949' #[86,128:-192,64:-64]
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-413\07-12-2010-50110\0.000000-67529' #[80,160:-192,92:-64]
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-412\06-26-2010-48381\0.000000-16672' #[98,160:-160,92:-64]
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-411\06-28-2010-76735\0.000000-10595' #[86,160:-160,92:-64]
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-410\07-08-2010-60028\0.000000-33774' #[86,160:-128,32:-32]
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-409\06-17-2010-58404\0.000000-18868' #[100,160:-160,64:-64]
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-408\06-18-2010-12138\0.000000-71178' #[73,160:-160,96:-96]
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-407\06-18-2010-48049\0.000000-73826' #[98,160:-160,96:-96]
#path = r'H:\Data\Lung1\Lung1 dicom\LUNG1-406\06-13-2010-92766\0.000000-68645' #[66,160:-128,96:-96]

#path = r'H:\Data\RIDER Lung CT\RIDER-1129164940\09-20-2006-1-96508\4.000000-24533' #[86,64:-32,:]
#path = r'H:\Data\RIDER Lung CT\RIDER-9763310455\09-16-2006-34385\100.000000-43945'
#path = r'H:\Data\RIDER Lung CT\RIDER-9762593735\02-21-2007-69768\102.000000-36948'
#path = r'H:\Data\RIDER Lung CT\RIDER-2016615262\02-06-2007-37305\103.000000-91150'
#path = r'H:\Data\RIDER Lung CT\RIDER-1760553574\04-23-2007-41616\12.000000-36105'
#path = r'H:\Data\RIDER Lung CT\RIDER-1532432635\04-12-2007-05084\109.000000-90930'
path = r'H:\Data\RIDER Lung CT\RIDER-2655999012\01-04-2007-1-37917\302.000000-70095'

scan = load_scan(path)
print('Initial image details:', scan.shape, scan.min(), scan.max())
plt.hist(scan.flatten(), bins=256)

scan[scan < -1024] = -1024
scan[scan > 3071] = 3071
if scan.max() < 3071:
    scan[:,0,0] = 3071
scan = normalize(scan)
#imgs = np.expand_dims(imgs, 0)
img_size = scan.shape[1:]

imgs = scan

print('All scan shape:    ', scan.shape)
print('The shape from the scan:', scan.shape[0])
print('Mins and maxes:', scan.min(), scan.max())

# Grab the selected slice
position = 4
key = 'RIDER-2655999012_2_4'
n = (scan.shape[0]//12) * position
imgs = scan[n-1:n+2, :, :]

imgs.shape

#imgs = imgs[:,64:-32,:]

plt.figure(figsize=(12,12))
plt.imshow(imgs[1,:,:].squeeze())
plt.show()

#imgs = np.moveaxis(imgs, 0, -1)
print('shape:', imgs.shape)
windows = get_windows(imgs)
windows = np.moveaxis(windows, 2, -1)

print('Shape of the windows:', windows.shape)

'''
fig, axs = plt.subplots(windows.shape[0], windows.shape[1], sharex='col', sharey='row',
                        gridspec_kw={'hspace':.02, 'wspace':.02}, figsize=(16, 16))
count = 0
for n in range(windows.shape[0]):
    for m in range(windows.shape[1]):
        axs[n, m].imshow(np.moveaxis(windows[n, m, :, :], 0, -1))
        count+=1
plt.show()
'''

#del model
gc.collect()
 
model = PixelCNNpp(args.image_dims, args.n_channels, args.n_res_layers, 
                   args.n_logistic_mix, args.n_cond_classes).to('cuda')

#model = torch.nn.DataParallel(model)

params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        params.append(name)

model_path = r'C:\Users\william\OneDrive\Desktop'
#model_file = 'checkpoint_112.pt'
model_file = 'C2_checkpoint_52.pt'
print('loading model ...', model_file)
#model_checkpoint = torch.load(args.restore_file, map_location=args.device)
model_checkpoint = torch.load(os.path.join(model_path, model_file), map_location='cpu')
state_dict = model_checkpoint['state_dict']
model = torch.nn.DataParallel(model).to(args.device)
model.load_state_dict(state_dict)

windows.shape


for i in range(windows.shape[0]):
    plt.figure(figsize=(16,16))
    for j in range(windows.shape[1]):
        test = windows[i, j, :, :]
        plt.subplot(1,15,j+1)
        plt.imshow(test[:, :, 1])
    plt.show()


''' Do the first window '''
x = windows[0, 0, :, :, :]
x.shape

print(x.min(), x.max())
#x = normalize256(x)
print(x.min(), x.max())

#x = ((x/255.)*2.)-1.

#x = Image.fromarray(x)
#transform = transforms.Compose([ transforms.ToTensor(),
#                                 transforms.Normalize(mean=[.5, .5, .5],std=[.5, .5, .5]) ])
#x = transform(x)

x = torch.from_numpy(x.astype(np.float32))
x.dtype
x.size()
x = x.permute(2, 0, 1)
x = x.unsqueeze(0)
x = x.unsqueeze(0)
x.size()
print(x.min(), x.max())

'''
plt.figure(figsize=(18, 18))
for i in range(3):
    plt.subplot(1,3,i+1)
    temp = x[:, :, i].detach().cpu().numpy().squeeze()
    plt.imshow(temp, cmap='gray')
plt.show()
'''
'''
logits = model(x)
output = sample_from_discretized_mix_logistic_1d(logits)
output.shape
plt.imshow(output.detach().cpu().numpy().squeeze())
'''

out = generate3d(x)
out.shape
#print(out[:,:,0].min(), out[:,:,0].max())
#print(out[:,:,1].min(), out[:,:,1].max())
#print(out[:,:,2].min(), out[:,:,2].max())

#plt.hist(out[:,:,0].detach().cpu().numpy().flatten(), bins=256)
#plt.hist(out[:,:,1].detach().cpu().numpy().flatten(), bins=256)
#plt.hist(out[:,:,2].detach().cpu().numpy().flatten(), bins=256)

#out[:, :, 1, 2:, :] = 0

plt.figure(figsize=(18, 18))
for i in range(3):
    plt.subplot(1,3,i+1)
    temp = out[:, :, i].detach().cpu().numpy().squeeze()
    plt.imshow(temp, cmap='gray')
plt.show()

###############################################################################

out.size()
out_img = out.detach().cpu().numpy().squeeze()
out_img.shape
plt.figure(figsize=(8, 8))
plt.imshow(out_img[1].squeeze())
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(x[:,:,1, :, :].squeeze())
plt.show()

#np.save('First.npy', out)

''' Do the first row and column windows '''

#whole_img = np.empty(img_size, dtype=np.float32)
whole_img = np.kron([[-1, 0] * 10, [0, -1] * 10] * 10, np.ones((32, 32)))[:32*(windows.shape[0]+1), :32*(windows.shape[1]+1)]
whole_img.shape
imgs.shape
#whole_img[:2, :] = scan[1,:2,:192]
plt.imshow(whole_img)
plt.show()
whole_img[:64, :64] = out_img[1].copy()
plt.imshow(whole_img)
plt.show()

windows.shape

for n in range(windows.shape[1]):
    if n > 0:
        x = windows[0, n, :, :, :].copy()
        x[:, :, 1] = get_patch(whole_img, (0,n)).clone().detach()
        x = np.rollaxis(x, -1)
        x = np.expand_dims(x, 0)
        x = np.expand_dims(x, 0)
        x = x.astype(np.float32)
        out = generate3d(x.copy(), xmin=32, convert=True, erase=False)
                
        whole_img = update_whole(whole_img, out.squeeze()[1, :, :], window=(0,n))
        plt.figure(figsize=(12, 12))
        plt.imshow(whole_img)
        plt.show()

np.save('whole_top', whole_img)
whole_img.shape
test = whole_img[:64, :288]
plt.figure(figsize=(18,18))
plt.imshow(test)

for n in range(windows.shape[0]):
    if n > 0:
        x = windows[n, 0, :, :, :].copy()
        x[:, :, 1] = get_patch(whole_img, (n,0)).clone().detach()
        x = np.rollaxis(x, -1)
        x = np.expand_dims(x, 0)
        x = np.expand_dims(x, 0)
        x = x.astype(np.float32)        
        out = generate3d(x.copy(), ymin=32, convert=True, erase=False)
                
        whole_img = update_whole(whole_img, out.squeeze()[1, :, :], window=(n,0))        
        plt.figure(figsize=(12, 12))
        plt.imshow(whole_img)
        plt.show()

imgs[1].min()
imgs[1].max()

        
np.save('whole_sides.npy', whole_img)

whole_img[-1,-1] =.5

plt.figure(figsize=(12,12))
plt.imshow(whole_img)

''' Do the internal windows '''
#whole_img = np.load('sides.npy')

for n in range(1, windows.shape[0]):
    print(n, n)
    x = windows[n, n, :, :, :]
    x = np.rollaxis(x, -1)
    x = np.expand_dims(x, 0)
    x = np.expand_dims(x, 0)
    x[:, :, 1] = get_patch(whole_img, (n,n))
    out = generate3d(x, xmin=32, ymin=32, erase=False, convert=True)
    #print(out.shape)
    whole_img = update_whole(whole_img, out.squeeze()[1, :, :], window=(n,n))
    plt.figure(figsize=(12, 12))
    plt.imshow(whole_img)
    plt.show()
    for m in range(n, windows.shape[0]):
        if m > n:
            #print(n, m)
            #print(m, n)
            
            horizontal = windows[n, m, :, :, :]
            horizontal = np.rollaxis(horizontal, -1)
            vertical = windows[m, n, :, :, :]
            vertical = np.rollaxis(vertical, -1)
            
            #print(horizontal.size(), vertical.size())
            
            x = np.stack([horizontal, vertical], axis=0)
            x = np.expand_dims(x, 1)
            #print(x.size())
    
            x[0, :, 1, :, :] = get_patch(whole_img, (n,m))
            x[1, :, 1, :, :] = get_patch(whole_img, (m,n))
            
            #print(x.size())
    
            out = generate3d(x, xmin=32, ymin=32, erase=False, convert=True)
            
            whole_img = update_whole(whole_img, out[0, :, 1, :, :], window=(n,m))
            whole_img = update_whole(whole_img, out[1, :, 1, :, :], window=(m,n))
            
            plt.imshow(whole_img)
            plt.show()
    
#np.save('whole_semi', whole_img)
plt.figure(figsize=(12, 12))
plt.imshow(whole_img)
np.save(key, whole_img)

plt.figure(figsize=(12, 12))
plt.imshow(imgs[1])
np.save(key[:16]+'_tru', imgs[1])

'''
for n in range(1, windows.shape[0]):
    if n > 1:break
    #print('N:',n, n++windows.shape[0]-1)
    x = windows[n, n+windows.shape[0], :, :, :]
    x = np.rollaxis(x, -1)
    x = np.expand_dims(x, 0)
    x = np.expand_dims(x, 0)
    x[:, :, 1] = get_patch(whole_img, (n,n))
    out = generate3d(x, xmin=32, ymin=32, erase=False, convert=True)
    #print(out.shape)
    whole_img = update_whole(whole_img, out.squeeze()[1, :, :], window=(n,n))
    plt.figure(figsize=(12, 12))
    plt.imshow(whole_img)
    plt.show()
    for m in range(n, windows.shape[1]):
        if m > n:
            #print('couple:',n, m+windows.shape[0]-1)
            #print('couple:',m, n+windows.shape[0]-1)

            horizontal = windows[n, m, :, :, :]
            horizontal = np.rollaxis(horizontal, -1)
            vertical = windows[m, n, :, :, :]
            vertical = np.rollaxis(vertical, -1)
            
            #print(horizontal.size(), vertical.size())
            
            x = np.stack([horizontal, vertical], axis=0)
            x = np.expand_dims(x, 1)
            #print(x.size())
    
            x[0, :, 1, :, :] = get_patch(whole_img, (n,m))
            x[1, :, 1, :, :] = get_patch(whole_img, (m,n))
            
            #print(x.size())
    
            out = generate3d(x, xmin=32, ymin=32, erase=False, convert=True)
            
            whole_img = update_whole(whole_img, out[0, :, 1, :, :], window=(n,m))
            whole_img = update_whole(whole_img, out[1, :, 1, :, :], window=(m,n))
            
            plt.imshow(whole_img)
            plt.show()
'''


