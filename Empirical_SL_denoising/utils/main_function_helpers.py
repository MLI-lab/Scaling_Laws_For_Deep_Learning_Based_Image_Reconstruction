import torch
import argparse
import os
import yaml
import pathlib
import pickle
import logging
import sys
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision
import glob
from torch.serialization import default_restore_location
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import utils
import models 

from utils.data_helpers.load_datasets_helpers import *
from utils.meters import *
from utils.progress_bar import *
from utils.noise_model import get_noise
from utils.metrics import ssim,psnr
from utils.test_metrics import *



def load_model(args):
    USE_CUDA = True
    device = torch.device('cuda') if (torch.cuda.is_available() and USE_CUDA) else torch.device('cpu')
    
    checkpoint_path = glob.glob(args.output_dir +'/unet*')
    if len(checkpoint_path) != 1:
        raise ValueError("There is either no or more than one model to load")
    checkpoint_path = pathlib.Path(checkpoint_path[0] + f"/checkpoints/checkpoint_{args.restore_mode}.pt")
    state_dict = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, "cpu"))
    args = argparse.Namespace(**{ **vars(state_dict["args"]), "no_log": True})


    model = models.unet_fastMRI(
            in_chans=args.in_chans,
            chans = args.chans,
            num_pool_layers = args.num_pool_layers,
            drop_prob = 0.0,
            residual_connection = args.residual,
        ).to(device)
    model.load_state_dict(state_dict["model"][0])
    model.eval()
    return model

def cli_main_test(args):
    USE_CUDA = True
    device = torch.device('cuda') if (torch.cuda.is_available() and USE_CUDA) else torch.device('cpu')
    
    model = load_model(args)
    
    # evaluate test performance over following noise range
    noise_std_range = np.linspace(args.test_noise_std_min, args.test_noise_std_max, 
                                  ((args.test_noise_std_max-args.test_noise_std_min)//args.test_noise_stepsize)+1,dtype=int)/255.
    
    metrics_path = os.path.join(args.output_dir, args.test_mode + '_' + str(args.test_noise_std_min)+'-'+str(args.test_noise_std_max)+f'_metrics_{args.restore_mode}.p')
    
    metrics_dict = metrics_avg_on_noise_range(model, args, noise_std_range, device = device)
    pickle.dump( metrics_dict, open(metrics_path, "wb" ) )

def cli_main(args):
    available_models = glob.glob(f'{args.output_dir}/*')
        
    if not args.resume_training and available_models:
        raise ValueError('There exists already a trained model and resume_training is set False')
    if args.resume_training: 
        f_restore_file(args)
        
    # reset the attributes of the function save_checkpoint
    mode = "max"
    default_score = float("inf") if mode == "min" else float("-inf")
    utils.save_checkpoint.best_score =  default_score
    utils.save_checkpoint.best_step = -1
    utils.save_checkpoint.best_epoch = -1
    utils.save_checkpoint.last_step = -1
    utils.save_checkpoint.current_lr = args.lr
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Set the name of the directory for saving results 
    utils.setup_experiment(args)
    
    utils.init_logging(args)
    
    # Build data loaders, model and optimizer
    model = models.unet_fastMRI(
            in_chans=args.in_chans,
            chans = args.chans,
            num_pool_layers = args.num_pool_layers,
            drop_prob = 0.0,
            residual_connection = args.residual,
        ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_gamma, patience=args.lr_patience, 
        threshold=args.lr_threshold, threshold_mode='abs', cooldown=0, 
        min_lr=args.lr_min, eps=1e-08, verbose=True
    )
    
    
    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    trainset = ImagenetSubdataset(args.train_size,args.path_to_ImageNet_train)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, generator=torch.Generator().manual_seed(args.seed))
    
    valset = ImagenetSubdataset(args.val_size,args.path_to_ImageNet_train,mode='val')
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, generator=torch.Generator().manual_seed(args.seed))
    
    if args.resume_training:
        state_dict = utils.load_checkpoint(args, model, optimizer, scheduler)
        global_step = state_dict['last_step']
        start_epoch = int(state_dict['last_step']/(len(train_loader)))+1
        start_decay = True
    else:
        global_step = -1
        start_epoch = 0
        if args.lr_annealing:
            start_decay = False
        else:
            start_decay = True
        
    args.log_interval = min(len(trainset), 100) # len(train_loader)=log once per epoch
    args.no_visual = False # True for not logging to tensorboard
    
    train_meters = {name: RunningAverageMeter(0.98) for name in (["train_loss"])}
    valid_meters = {name: AverageMeter() for name in (["valid_psnr", "valid_ssim"])}
    writer = SummaryWriter(log_dir=args.experiment_dir) if not args.no_visual else None
    
    break_counter = 0
    for epoch in range(start_epoch, args.num_epochs):
        start = time.process_time()
        train_bar = ProgressBar(train_loader, epoch)

        for meter in train_meters.values():
            meter.reset()

        for batch_id, inputs in enumerate(train_bar):
            model.train() 

            global_step += 1
            inputs = inputs.to(device)
            noise = get_noise(inputs, noise_std = args.noise_std/255.)

            noisy_inputs = noise + inputs;
            outputs = model(noisy_inputs)
            loss = F.mse_loss(outputs, inputs, reduction="sum") / torch.prod(torch.tensor(inputs.size()))

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_meters["train_loss"].update(loss.item())
            train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)

            if writer is not None and global_step % args.log_interval == 0:
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("loss/train", loss.item(), global_step)
                sys.stdout.flush()
                
            

        if epoch % args.valid_interval == 0:
            model.eval()
            gen_val = torch.Generator()
            gen_val = gen_val.manual_seed(10)
            for meter in valid_meters.values():
                meter.reset()

            valid_bar = ProgressBar(val_loader)
            for sample_id, sample in enumerate(valid_bar):

                with torch.no_grad():
                    sample = sample.to(device)
                    noise =  torch.randn(sample.shape,generator = gen_val) * args.noise_std/255.

                    noisy_inputs = noise.to(device) + sample;
                    output = model(noisy_inputs)
                    valid_psnr = psnr(output, sample)
                    valid_meters["valid_psnr"].update(valid_psnr.item())
                    valid_ssim = ssim(output, sample)
                    valid_meters["valid_ssim"].update(valid_ssim.item())
                    
            if writer is not None:
                writer.add_scalar("psnr/valid", valid_meters['valid_psnr'].avg, global_step)
                writer.add_scalar("ssim/valid", valid_meters['valid_ssim'].avg, global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
                sys.stdout.flush()
            
            if utils.save_checkpoint.best_score < valid_meters["valid_psnr"].avg and not start_decay:                
                utils.save_checkpoint(args, global_step, epoch, model, optimizer, score=valid_meters["valid_psnr"].avg, mode="max")
                current_lr = utils.save_checkpoint.current_lr
                optimizer.param_groups[0]["lr"] = current_lr*args.lr_beta
                utils.save_checkpoint.current_lr = current_lr*args.lr_beta
                annealing_counter = 0
            elif not start_decay:
                annealing_counter += 1
                current_lr = utils.save_checkpoint.current_lr
                if annealing_counter == args.lr_patience_annealing:
                    
                    available_models = glob.glob(f'{args.output_dir}/*')
                    if not available_models:
                        raise ValueError('No file to restore')
                    elif len(available_models)>1:
                        raise ValueError('Too many files to restore from')
                        
                    model_path = os.path.join(available_models[0], "checkpoints/checkpoint_best.pt")    
                    state_dict = torch.load(model_path, map_location=lambda s, l: default_restore_location(s, "cpu"))
                    model = [model] if model is not None and not isinstance(model, list) else model
                    for m, state in zip(model, state_dict["model"]):
                        m.load_state_dict(state)
                    model = model[0]
                    
                    optimizer.param_groups[0]["lr"] = current_lr/args.lr_beta
                    start_decay = True
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='max', factor=args.lr_gamma, patience=args.lr_patience, 
                        threshold=args.lr_threshold, threshold_mode='abs', cooldown=0, 
                        min_lr=args.lr_min, eps=1e-08, verbose=True
                    )
            else:
                utils.save_checkpoint(args, global_step, epoch, model, optimizer, score=valid_meters["valid_psnr"].avg, mode="max")
                current_lr = optimizer.param_groups[0]["lr"]
                
            
        
        if writer is not None:            
            writer.add_scalar("epoch", epoch, global_step)
            sys.stdout.flush()
        
        if start_decay:
            scheduler.step(valid_meters['valid_psnr'].avg)   
            
        end = time.process_time() - start
        logging.info(train_bar.print(dict(**train_meters, **valid_meters, lr=current_lr, time=np.round(end/60,3))))
        
        if optimizer.param_groups[0]["lr"] == args.lr_min and start_decay:
            break_counter += 1
        if break_counter == args.break_counter:
            print('Break training due to minimal learning rate constraing!')
            break

    logging.info(f"Done training! Best PSNR {utils.save_checkpoint.best_score:.3f} obtained after step {utils.save_checkpoint.best_step} (epoch {utils.save_checkpoint.best_epoch}).")




def get_args(hp,rr):
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--train-size", default=None, help="number of examples in training set")
    parser.add_argument("--val-size", default=40, help="number of examples in validation set")
    parser.add_argument("--test-size", default=100, help="number of examples in test set")
    
    parser.add_argument("--batch-size", default=128, type=int, help="train batch size")

    # Add model arguments
    parser.add_argument("--model", default="unet", help="model architecture")

    # Add noise arguments
    parser.add_argument('--noise_std', default = 15, type = float, 
                help = 'noise level')
    parser.add_argument('--test_noise_std_min', default = 15, type = float, 
                help = 'minimal noise level for testing')
    parser.add_argument('--test_noise_std_max', default = 15, type = float, 
                help = 'maximal noise level for testing')
    parser.add_argument('--test_noise_stepsize', default = 5, type = float, 
                help = 'Stepsize between test_noise_std_min and test_noise_std_max')

    # Add optimization arguments
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--lr-gamma", default=0.5, type=float, help="factor by which to reduce learning rate")
    parser.add_argument("--lr-patience", default=5, type=int, help="epchs without improvement before lr decay")
    parser.add_argument("--lr-min", default=1e-5, type=float, help="Once we reach this learning rate continue for break_counter many epochs then stop.")
    parser.add_argument("--lr-threshold", default=0.003, type=float, help="Improvements by less than this threshold are not counted for decay patience.")
    parser.add_argument("--break-counter", default=9, type=int, help="Once smallest learning rate is reached, continue for so many epochs before stopping.")
    parser.add_argument("--num-epochs", default=100, type=int, help="force stop training at specified epoch")
    parser.add_argument("--valid-interval", default=1, type=int, help="evaluate every N epochs")
    parser.add_argument("--save-interval", default=1, type=int, help="save a checkpoint every N steps")

    # Add model arguments
    parser = models.unet_fastMRI.add_args(parser)
    
    parser = utils.add_logging_arguments(parser)

    args, _ = parser.parse_known_args()
    
    # Set arguments specific for this experiment
    dargs = vars(args)
    for key in hp.keys():
        dargs[key] = hp[key][0]
    args.seed = int(42 + 10*rr)
    
    return args


def f_restore_file(args):
    available_models = glob.glob(f'{args.output_dir}/*')
    if not available_models:
        raise ValueError('No file to restore')
    if not args.restore_mode:
        raise ValueError("Pick restore mode either 'best' 'last' or '\path\to\checkpoint\dir'")
    if args.restore_mode=='best':
        mode = "max"
        best_score = float("inf") if mode == "min" else float("-inf")
        best_model = None
        for modelp in available_models:
            model_path = os.path.join(modelp, "checkpoints/checkpoint_best.pt")
            if os.path.isfile(model_path):
                state_dict = torch.load(model_path, map_location=lambda s, l: default_restore_location(s, "cpu"))
                score = state_dict["best_score"]
                if (score < best_score and mode == "min") or (score > best_score and mode == "max"):
                    best_score = score
                    best_model = model_path
                    best_modelp = modelp
                    best_step = state_dict["best_step"]
                    best_epoch = state_dict["best_epoch"]
        args.restore_file = best_model
        args.experiment_dir = best_modelp

    elif args.restore_mode=='last':
        last_step = -1
        last_model = None
        for modelp in available_models:
            model_path = os.path.join(modelp, "checkpoints/checkpoint_last.pt")
            if os.path.isfile(model_path):
                state_dict = torch.load(model_path, map_location=lambda s, l: default_restore_location(s, "cpu"))
                step = state_dict["last_step"]
                if step > last_step:
                    last_step = step
                    last_model = model_path
                    last_modelp = modelp
                    score = state_dict["score"]
                    last_epoch = state_dict["epoch"]
        args.restore_file = last_model
        args.experiment_dir = last_modelp

    else:
        args.restore_file = args.restore_mode
        args.experiment_dir = args.restore_mode[:args.restore_mode.find('/checkpoints')]
            
    