import os
import json
import numpy as np

from fastmri.main_functions_helpers import *

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
        '--testing',
        choices=(True, False),
        default=True,
        type=bool,
        help='Evaluate best and last checkpoint on validation and test set.'
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
        '--path_to_fastMRI_brain_dataset',
        type=str,
        help='Path to fastMRI brain directory containing both the training and validation set.',
        required=True
    )
    
    command_line_args = parser.parse_args()
    
    hps = []
    for train_size,channel in zip(command_line_args.train_sizes, command_line_args.channels):
        options_name = "options/trainsize{}_channels{}.txt".format(train_size,channel)

        # Load hyperparameter options
        with open(options_name) as handle:
            hp = json.load(handle)
        hps.append(hp)
    
    
    for ee in range(len(command_line_args.exp_nums)):

        hp = hps[ee]
        num_runs = list(np.arange(hp['num_runs'][0]))
        for rr in num_runs:
            exp_name =  'E' + command_line_args.exp_nums[ee] + \
                        '_t' + str(hp['num_examples'][0]) + \
                        '_l' + '4' + \
                        'c' + str(hp['chans'][0]) + \
                        '_bs' + '1' +\
                        '_lr' + '001'
            if rr>0:
                exp_name = exp_name + '_run{}'.format(rr+1)
            if not os.path.isdir('./'+exp_name):
                os.mkdir('./'+exp_name)
            create_fastmri_dirs_yaml(command_line_args.path_to_fastMRI_brain_dataset,exp_name)

            ########
            # Training
            ########
            if command_line_args.training:  
                print('\n{} - Training\n'.format(exp_name))
                args = build_args(hp,rr)
                cli_main(args)
                print('\n{} - Training finished\n'.format(exp_name))

            ########
            # Testing
            ########
            if command_line_args.testing:
                print('\n{} - Testing\n'.format(exp_name))
                test_modes = ["test_on_val","test_on_test"]

                for test_mode in test_modes:
                    for resume_from_which_checkpoint in ["last","best"]:

                        args = build_args(hp,rr,test_mode)
                        args.mode = "test"
                        args.logger = False
                        args.test_path=args.data_path/"multicoil_val"
                        cli_main(args)
                        if test_mode == "test_on_test" or test_mode == "test_on_val":
                            tm = test_mode[8:]
                        else:
                            tm = test_mode
                        metrics_filename = './'+exp_name+'/log_files/metrics_'+exp_name+'_{}_{}.pkl'.format(tm,resume_from_which_checkpoint)
                        if resume_from_which_checkpoint=="best":
                            ckpt = args.resume_from_checkpoint
                            ind1 = str(ckpt).find('epoch=')
                            ind2 = str(ckpt).find('-step')
                            epoch = str(ckpt)[ind1+len('epoch='):ind2]
                            metrics_filename = metrics_filename[:-4]+'_'+epoch+'ep'+metrics_filename[-4:]
                        evaluate_reconstructions(test_mode,metrics_filename)

                print('\n{} - Testing finished\n'.format(exp_name)) 
                
if __name__ == '__main__':
    run() 
    