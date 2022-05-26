import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
#import cv2
from utils.noise_model import get_noise
from utils.metrics import ssim,psnr
from skimage import color
import PIL.Image as Image
import torchvision.transforms as transforms

metrics_key = ['psnr_m', 'psnr_s', 'psnr_delta_m', 'psnr_delta_s', 'ssim_m', 'ssim_s', 'ssim_delta_m', 'ssim_delta_s'];

def tensor_to_image(torch_image, low=0.0, high = 1.0, clamp = True):
    if clamp:
        torch_image = torch.clamp(torch_image, low, high);
    return torch_image[0,0].cpu().data.numpy()


def normalize(data):
    return data/255.

def convert_dict_to_string(metrics):
    return_string = '';
    for x in metrics.keys():
        return_string += x+': '+str(round(metrics[x], 3))+' ';
    return return_string



def get_all_comparison_metrics(denoised, source, noisy = None,  return_title_string = False, clamp = True):

    

    metrics = {};
    metrics['psnr'] = np.zeros(len(denoised))
    metrics['ssim'] = np.zeros(len(denoised))
    if noisy is not None:
        metrics['psnr_delta'] = np.zeros(len(denoised))
        metrics['ssim_delta'] = np.zeros(len(denoised))

    if clamp:
        denoised = torch.clamp(denoised, 0.0, 1.0)


    metrics['psnr'] = psnr(source, denoised);
    metrics['ssim'] = ssim(source, denoised);

    if noisy is not None:
        metrics['psnr_delta'] = metrics['psnr'] - psnr(source, noisy);
        metrics['ssim_delta'] = metrics['ssim'] - ssim(source, noisy);




    if return_title_string:
        return convert_dict_to_string(metrics)
    else:
        return metrics


def average_on_folder(args, net, noise_std, 
            verbose=True, device = torch.device('cuda')):
    
    print(f'\n Dataset: {args.test_mode}, Restore mode: {args.restore_mode}')
    load_path = './training_set_lists/'
    
    seed_dict = {
        "val":10,
        "test":20
    }
    gen = torch.Generator()
    gen = gen.manual_seed(seed_dict[args.test_mode])
    
    if args.test_mode == 'test':
        files_source = torch.load(load_path+f'ImageNetTest{args.test_size}_filepaths.pt')        
    elif args.test_mode == 'val':
        files_source = torch.load(load_path+f'ImageNetVal{args.val_size}_filepaths.pt')      
    
    avreage_metrics_key = ['psnr', 'psnr_delta', 'ssim', 'ssim_delta']
    avg_metrics = {};
    for x in avreage_metrics_key:
        avg_metrics[x] = [];
    
    psnr_list = []
    ssim_list = []
    
    for f in files_source:
        transformT = transforms.ToTensor()
        ISource = torch.unsqueeze(transformT(Image.open(args.path_to_ImageNet_train + f).convert("RGB")),0).to(device)
        noise =  torch.randn(ISource.shape,generator = gen) * args.noise_std/255.

        INoisy = noise.to(device) + ISource;
        
        out = torch.clamp(net(INoisy), 0., 1.)

        ind_metrics = get_all_comparison_metrics(out, ISource, INoisy, return_title_string = False);

        for x in avreage_metrics_key:
            avg_metrics[x].append(ind_metrics[x])

        if(verbose):
            print("%s %s" % (f, convert_dict_to_string(ind_metrics)))
    
    metrics = {}
    for x in avreage_metrics_key:
        metrics[x+'_m'] = np.mean(avg_metrics[x])
        metrics[x+'_s'] = np.std(avg_metrics[x])

    if verbose:
        print("\n Average %s" % (convert_dict_to_string(metrics)))

    return metrics


def metrics_avg_on_noise_range(net, args,noise_std_array, device = torch.device('cuda')):
    array_metrics = {};
    for x in metrics_key:
        array_metrics[x] = np.zeros(len(noise_std_array))

    for j, noise_std in enumerate(noise_std_array):
        metric_list = average_on_folder(args, net, 
                                                noise_std = noise_std,
                                                verbose=False, device=device);

        for x in metrics_key:
            array_metrics[x][j] += metric_list[x]
            print('noise: ', int(noise_std*255), ' ', x, ': ', str(array_metrics[x][j]))

    return array_metrics

