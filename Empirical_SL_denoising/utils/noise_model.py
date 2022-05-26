import torch

def get_noise(data, noise_std = float(25)/255.0):
    noise = torch.randn_like(data);
    noise.data = noise.data * noise_std;
    return noise
