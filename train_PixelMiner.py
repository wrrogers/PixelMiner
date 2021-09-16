import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST, CIFAR10
from torchvision import datasets, transforms

import numpy as np
from tqdm import tqdm

import os
import pickle
import time
import json
import pprint
from functools import partial
import argparse

import wandb

from torch.optim import Adam, RMSprop, SGD

from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import mean_squared_error as MSE

from pixelcnnpp import sample_from_discretized_mix_logistic, PSNR

from parameters import Parameters

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#torch.cuda.empty_cache()

# --------------------
# Data
# --------------------

def fetch_dataloaders(args):
    # preprocessing transforms
    transform = T.Compose([
                            T.ToTensor(),                                            # tensor in [0,1]
                            transforms.Normalize(mean=[0, 0, 0],std=[1, 1, 1]),
                            lambda x: x.mul(255).div(2**(8-args.n_bits)).floor(),    # lower bits (Quantize)
                            partial(preprocess, n_bits=args.n_bits)                  # to model space [-1,1]
                          ])

    #target_transform = (lambda y: torch.eye(args.n_cond_classes)[y]) if args.n_cond_classes else None

    if args.dataset=='lung':
        from lung_dataset import LungDataset
        train_dataset = LungDataset((64, 64), crop = True, split=True, transform=transform, train=True)
        valid_dataset = LungDataset((64, 64), crop = True, split=True, transform=transform, train=False)
    else:
        raise RuntimeError('Dataset not recognized')

    print('\n\nDataset:',args.dataset)
    print('Traing length:', len(train_dataset))
    print('Test length:', len(valid_dataset), '\n\n')

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=(args.device.type=='cuda'), num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, pin_memory=(args.device.type=='cuda'), num_workers=0)

    # save a sample
    #data_sample = next(iter(train_dataloader))[0]
    #writer.add_image('data_sample', make_grid(data_sample, normalize=True, scale_each=True), args.step)
    #save_image(data_sample, os.path.join(args.output_dir, 'data_sample.png'), normalize=True, scale_each=True)

    return train_dataloader, valid_dataloader

def preprocess(x, n_bits):
    # 1. convert data to float
    # 2. normalize to [0,1] given quantization
    # 3. shift to [-1,1]
    return x.float().div(2**n_bits - 1).mul(args.scale_input).add(args.shift_input)

def deprocess(x, n_bits):
    # 1. shift to [0,1]
    # 2. quantize to n_bits
    # 3. convert data to long
    return x.add(1).div(2).mul(2**n_bits - 1).long()

def save_json(data, filename, args):
    with open(os.path.join(args.output_dir, filename + '.json'), 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# --------------------
# Train, evaluate, generate
# --------------------

def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, epoch, args):
    model.train()

    with tqdm(total=len(dataloader), desc='epoch {}/{}'.format(epoch, args.start_epoch + args.n_epochs)) as pbar:
        for x in dataloader:
            y = None
            args.step += 1

            #print(x.size())

            #x = ((x-x.min())/(x.max()-x.min()))
            #x = x*2
            #x = x-1

            x = x.to(args.device)
            logits = model(x, y.to(args.device) if args.n_cond_classes else None)
            loss = loss_fn(logits, x, args.n_bits).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()

            pbar.set_postfix(
                             mm='{:.4f}, {:.4f}'.format(x.min(), x.max()),
                             lr='{:.7f}'.format(optimizer.param_groups[0]['lr']),
                             l='{:.3f}, {:.3f}'.format(logits.min(), logits.max()),
                             loss='{:.4f}, {:.4f}'.format(loss.item(), loss.item() / (np.log(2) * np.prod(args.image_dims))),
                             #loss='{:.4f}'.format(loss.item() / (np.log(2) * np.prod(args.image_dims))),
                             )
            pbar.update()

            wandb.log({
                         "train_loss": loss.item() / (np.log(2) * np.prod(args.image_dims)),
                   })

            # record
            if args.step % args.log_interval == 0:
                # Additional Metrics
                sample = sample_from_discretized_mix_logistic(logits, args.image_dims)
                #x_sample = sample.detach().cpu().numpy()
                x_sample = ((sample.clone().detach().cpu().numpy() + 1) / 2) * 255
                #x_check = x.clone().detach().cpu().numpy()
                x_check  = ((x.clone().detach().cpu().numpy()      + 1) / 2) * 255

                if args.train_gen:
                    plt.figure(figsize=(10, 10))
                    for n in range(3):
                        plt.subplot(1,3,n+1)
                        plt.imshow(x_check[0,n,:,:])
                    plt.show()

                    plt.figure(figsize=(10, 10))
                    for n in range(3):
                        plt.subplot(1,3,n+1)
                        plt.imshow(x_sample[0,n,:,:])
                    plt.show()

                wdist = wasserstein_distance(x_check.flatten(), x_sample.flatten())
                psnr = PSNR(x_check, x_sample)

                x_check = np.moveaxis(x_check, 1, -1)
                x_sample = np.moveaxis(x_sample, 1, -1)

                ssims, mses = [], []
                for i in range(x_check.shape[0]):
                    ssims.append(SSIM(x_check[i], x_sample[i], data_range=2, multichannel=True))
                    mses.append(MSE(x_check[i], x_sample[i]))
                ssim = sum(ssims)/len(ssims)
                mse = sum(mses)/len(mses)

                print('\n\nWasserstien: {}, PSNR: {}, SSIM: {}, MSE: {}\n'.format(wdist, psnr, ssim, mse))

                wandb.log({
                            "wasserstein": wdist,
                            "psnr": psnr,
                            "ssim": ssim,
                            "mse": mse,
                   })

                #writer.add_scalar('train_bits_per_dim', loss.item() / (np.log(2) * np.prod(args.image_dims)), args.step)
                #writer.add_scalar('lr', optimizer.param_groups[0]['lr'], args.step)

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, args):
    model.eval()

    losses = 0
    for x in tqdm(dataloader, desc='Evaluate'):
        x = x.to(args.device)
        y = None
        logits = model(x, y.to(args.device) if args.n_cond_classes else None)
        losses += loss_fn(logits, x, args.n_bits).mean(0).item()
    return losses / len(dataloader)

@torch.no_grad()
def generate(model, data_loader, generate_fn, args):
    model.eval()
    if args.n_cond_classes:
        samples = []
        for h in range(args.n_cond_classes):
            h = torch.eye(args.n_cond_classes)[h,None].to(args.device)
            samples += [generate_fn(model, data_loader, args.n_samples, args.image_dims, args.device, h=h)]
        samples = torch.cat(samples)
    else:
        samples = generate_fn(model, data_loader, args.n_samples, args.image_dims, args.device)
        #print("SAMPLE INFO:", info)
    return make_grid(samples.cpu(), normalize=True, scale_each=True, nrow=args.n_samples)

def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, loss_fn, generate_fn, args):
    for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
        # train
        train_epoch(model, train_dataloader, optimizer, scheduler, loss_fn, epoch, args)

        if (epoch+1) % args.save_interval == 0:
              # save model
            torch.save({'epoch': epoch,
                        'global_step': args.step,
                        'state_dict': model.state_dict()},
                        os.path.join(args.output_dir, 'checkpoint_{}.pt'.format(epoch+1)))
            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, 'optim_checkpoint.pt'))
            if scheduler: torch.save(scheduler.state_dict(), os.path.join(args.output_dir, 'sched_checkpoint.pt'))


        if (epoch+1) % args.eval_interval == 0:
            # swap params to ema values
            #optimizer.swap_ema()

            # evaluate
            eval_loss = evaluate(model, test_dataloader, loss_fn, args)
            #print('Evaluate bits per dim: {:.3f}'.format(eval_loss.item() / (np.log(2) * np.prod(args.image_dims))))
            print('\n\nEvaluate bits per dim: {:.3f}\n'.format(eval_loss / (np.log(2) * np.prod(args.image_dims))))
            #writer.add_scalar('eval_bits_per_dim', eval_loss.item() / (np.log(2) * np.prod(args.image_dims)), args.step)

            wandb.log({
                         "test_loss": eval_loss/ (np.log(2) * np.prod(args.image_dims)),
                      })

        if (epoch+1) % args.gen_interval == 0:
            # generate
            samples = generate(model, test_dataloader, generate_fn, args)
            #writer.add_image('samples', samples, args.step)
            print(samples.size())
            for z in range(3):
                save_image(samples[z], os.path.join(args.output_dir, 'generation_sample_step_{}_{}.png'.format(args.step, z+1)))

            # restore params to gradient optimized
            #optimizer.swap_ema()

# --------------------
# Main
# --------------------

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    print('Visible Device:', os.environ['CUDA_VISIBLE_DEVICES'])

    params = Parameters()
    parser = argparse.ArgumentParser()

    hyper_parameter_defaults = dict(
            optimizer      = params.optimizer,
            lr             = params.lr,
            decay          = params.decay,
            #lr_decay       = params.lr_decay,
            #init           = 'normal',
            drop_rate      = params.drop_rate,
            eta_min        = params.eta_min,
            #n_logistic_mix = 10
            )

    wandb.init(config=hyper_parameter_defaults, project='Lung1 - v5')
    config = wandb.config

    for param in params.__dict__.items():
        if not callable(param[1]):
            print('--{}'.format(param[0]), type(param[1]), param[1])
            parser.add_argument('--{}'.format(param[0]), default=param[1])

    args = parser.parse_args()

    wandb.config.update(args, allow_val_change=True)

    args.output_dir = os.path.dirname(args.restore_file) if args.restore_file else \
                        os.path.join(os.getcwd(), 'results', args.model, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))

    #writer = SummaryWriter(log_dir = args.output_dir)

    # save config
    if not os.path.exists(args.output_dir): os.mkdir(args.output_dir)
    #if not os.path.exists(os.path.join(args.output_dir, 'config.json')): save_json(args.__dict__, 'config', args)
    #writer.add_text('config', str(args.__dict__))
    #pprint.pprint(args.__dict__)

    args.device = torch.device('cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_dataloader, test_dataloader = fetch_dataloaders(args)

    import pixelcnnpp
    model = pixelcnnpp.PixelCNNpp(args.image_dims, args.n_channels, args.n_res_layers, args.n_logistic_mix,
                                  args.n_cond_classes, drop_rate=float(config.drop_rate)).to(args.device)
    if not args.restore_file:
        model = torch.nn.DataParallel(model).to(args.device)
    loss_fn = pixelcnnpp.loss_fn
    generate_fn = pixelcnnpp.generate_fn

    if config.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=float(config.lr), betas=(float(config.decay), 0.9995))
    elif config.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=float(config.lr), momentum=float(config.decay))

    if args.scheduler == 'cosine_annealing_wr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 512, T_mult=2, eta_min=0.0006, last_epoch=-1)
    elif args.scheduler == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 512, eta_min=float(config.eta_min), last_epoch=-1)
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, float(config.lr_decay))

    #    print(model)
    print('Model parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

    if args.restore_file:
        print('loading model ...')
        #model_checkpoint = torch.load(args.restore_file, map_location=args.device)
        model_checkpoint = torch.load(args.restore_file, map_location='cpu')
        state_dict = model_checkpoint['state_dict']
        #state_dict.keys()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model = torch.nn.DataParallel(model).to(args.device)

        if args.restore_opt:
            print('loading optomizer ...')
            #optimizer.load_state_dict(torch.load(os.path.dirname(args.restore_file) + '/optim_checkpoint.pt', map_location=args.device))
            optimizer.load_state_dict(torch.load(os.path.dirname(args.restore_file) + '/optim_checkpoint.pt', map_location='cpu'))
            if scheduler:
                #scheduler.load_state_dict(torch.load(os.path.dirname(args.restore_file) + '/sched_checkpoint.pt', map_location=args.device))
                scheduler.load_state_dict(torch.load(os.path.dirname(args.restore_file) + '/sched_checkpoint.pt', map_location='cpu'))
            args.start_epoch = model_checkpoint['epoch'] + 1
            args.step = model_checkpoint['global_step']

    wandb.watch(model) # WATCH THE MODEL!

    if args.train:
        train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, loss_fn, generate_fn, args)

    if args.evaluate:
        #if args.step > 0: optimizer.swap_ema()
        eval_loss = evaluate(model, test_dataloader, loss_fn, args)
        print('Evaluate bits per dim: {:.3f}'.format(eval_loss / (np.log(2) * np.prod(args.image_dims))))
        #if args.step > 0: optimizer.swap_ema()
