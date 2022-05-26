import os
import numpy as np
import json
from argparse import ArgumentParser

from utils.main_function_helpers import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run():
    parser = ArgumentParser()
    
    parser.add_argument(
        '--training',
        choices=(True, False),
        default=True,
        type=bool,
        help='Start or continue training if there exists already a checkpoint.'
    )
    parser.add_argument(
        '--val_testing',
        choices=(True, False),
        default=True,
        type=bool,
        help='Evaluate best and last checkpoint on validation set.'
    )
    
    parser.add_argument(
        '--test_testing',
        choices=(True, False),
        default=True,
        type=bool,
        help='Evaluate best and last checkpoint on test set.'
    )

    parser.add_argument(
        '--exp_nums',
        nargs='+',
        help='Specify an ID for each experiment, e.g. 001,002,... ',
        required=True
    )
    
    parser.add_argument(
        '--train_sizes',
        nargs='+',
        help='The combintion of training set sizes and network sizes (channels) determine the experiments that are run. See options/ for available combinations.',
        required=True
    )
    
    parser.add_argument(
        '--channels',
        nargs='+',
        help='The combintion of training set sizes and network sizes (channels) determine the experiments that are run. See options/ for available combinations.',
        required=True
    )
    
    parser.add_argument(
        '--path_to_ImageNet_train',
        type=str,
        help='Path to ImageNet train directory.',
        required=True
    )
    
    command_line_args = parser.parse_args()
    
    hps = []
    for train_size,channel in zip(command_line_args.train_sizes, command_line_args.channels):
        options_name = "options/trainsize{}_channels{}.txt".format(train_size,channel)

        # Load hyperparameter options
        with open(options_name) as handle:
            hp = json.load(handle)
        hp['path_to_ImageNet_train'] = [command_line_args.path_to_ImageNet_train]
        hps.append(hp)
    
    for ee in range(len(command_line_args.exp_nums)):
    
        hp = hps[ee]
        num_runs = list(np.arange(hp['num_runs'][0]))

        for rr in num_runs:
            exp_name =  'E' + command_line_args.exp_nums[ee] + \
                        '_t' + str(hp['train_size'][0]) + \
                        '_l' + str(hp['num_pool_layers'][0]) + \
                        'c' + str(hp['chans'][0]) + \
                        '_bs' + str(hp['batch_size'][0]) +\
                        '_lr' + str(hp['lr'][0])[2:]
            if rr>0:
                exp_name = exp_name + '_run{}'.format(rr+1)
            if not os.path.isdir('./'+exp_name):
                os.mkdir('./'+exp_name)

            ########
            # Training
            ########
            if command_line_args.training:  
                print('\n{} - Training\n'.format(exp_name))
                args = get_args(hp,rr)
                args.output_dir = './'+exp_name
                cli_main(args)
                print('\n{} - Training finished\n'.format(exp_name))

            ########
            # Testing
            ########
            if command_line_args.val_testing or command_line_args.test_testing:
                print('\n{} - Testing\n'.format(exp_name))

                test_modes = []
                if command_line_args.val_testing:
                    test_modes.append("val")
                if command_line_args.test_testing:
                    test_modes.append("test")

                for test_mode in test_modes:
                    for restore_mode in ["last","best"]:
                        args = get_args(hp,rr)
                        args.output_dir = './'+exp_name
                        args.restore_mode = restore_mode
                        args.test_mode = test_mode
                        args.test_noise_std_min = args.noise_std
                        args.test_noise_std_max = args.noise_std
                        cli_main_test(args)

                print('\n{} - Testing finished\n'.format(exp_name))   
                
if __name__ == '__main__':
    run()            
    