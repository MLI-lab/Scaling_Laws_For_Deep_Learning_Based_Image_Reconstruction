import os
import pickle
from argparse import ArgumentParser
import pathlib
from fastmri.data.mri_data import fetch_dir
from fastmri.pl_modules import FastMriDataModule, UnetModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from fastmri.evaluate import *
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    train_transform = UnetDataTransform(args.challenge,mask_func=mask, use_seed=True)
    val_transform = UnetDataTransform(args.challenge,mask_func=mask, use_seed=True)
    test_transform = UnetDataTransform(args.challenge,mask_func=mask, use_seed=True)
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        combine_train_val=True, 
        test_split=args.test_split,
        test_path=args.test_path,
        test_mode=args.test_mode,
        use_filename_list=args.use_filename_list,
        filename_list=args.filename_list,
        sample_rate=args.sample_rate,
        volume_sample_rate=args.volume_sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator == "ddp"), 
    )
    
    # ------------
    # model
    # ------------
    if args.mode == "train":
        model = UnetModule(
            in_chans=args.in_chans,
            out_chans=args.out_chans,
            chans=args.chans,
            num_pool_layers=args.num_pool_layers,
            drop_prob=args.drop_prob,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
            lr_patience=args.lr_patience,
            lr_min=args.lr_min,
            lr_threshold=args.lr_threshold,
        )
    elif args.mode == "test":
        ####
        # Get latest checkpoint
        ####
        checkpoint_dir_general = args.default_root_dir / "lightning_logs"
        if checkpoint_dir_general.exists():
            args.resume_from_checkpoint = get_checkpoint(checkpoint_dir_general, args.resume_from_which_checkpoint)
            print('Resume from checkpoint {}'.format(args.resume_from_checkpoint))
        else:
            raise FileNotFoundError(
            'There exists no trained checkpoint to load for test inference.'
            )
        ####
        # load latest checkpoint
        ####
        model = UnetModule.load_from_checkpoint(args.resume_from_checkpoint) 
    else:
        raise ValueError(f"unrecognized mode {args.mode}") 
    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args(hp={},rr=0,test_mode="test_on_val"):
    filename_list = "train_"+str(hp['num_examples'][0])+"_filenames.yaml"
    resume_from_which_checkpoint = hp['resume_from_which_checkpoint'][0]
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("fastmri_dirs.yaml")
    backend = "dp" 
    num_gpus = 1
    batch_size = 1

    # set defaults based on optional directory config
    data_path = fetch_dir("brain_path", path_config)
    default_root_dir = (
        fetch_dir("log_path", path_config) 
    )

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )
    parser.add_argument(
        "--resume_from_which_checkpoint",
        default="last",
        choices=("last", "best"),
        type=str,
        help="last or best",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="equispaced",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08], 
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4], 
        type=int,
        help="Acceleration rates to use for masks",
    )

    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        mask_type="equispaced",  
        challenge="multicoil",  
        batch_size=batch_size,  
        test_path=None,  
        test_mode=test_mode, 
        use_filename_list=True, 
        filename_list=filename_list, # list with training images 
        volume_sample_rate=0.01, # we use filename_list instead
        num_workers = 8,        
        test_split = "test",
    )

    # module config
    parser = UnetModule.add_model_specific_args(parser)
    parser.set_defaults(
        in_chans=1,  # number of input channels to U-Net
        out_chans=1,  # number of output chanenls to U-Net
        chans=64,  # number of top-level U-Net channels
        num_pool_layers=4,  # number of U-Net pooling layers
        drop_prob=0.0,  # dropout probability
        lr=0.001,  # RMSProp learning rate
        lr_step_size=50,  # epoch at which to decrease learning rate (not used)
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight decay regularization strength
        lr_patience=5,
        lr_min=1e-6,
        lr_threshold=1e-4,
    )

    # trainer config
    checkpoint_callback_best = ModelCheckpoint(
        save_last=False,
        save_top_k=1,
        monitor="val_loss",
        filename="best-{epoch:02d}-{step:02d}-{val_loss:.6f}",
        mode="min",
    )
    checkpoint_callback_last = ModelCheckpoint(
        save_last=False,
        filename="last-{epoch:02d}-{step:02d}-{val_loss:.6f}",
    )
    
    
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  
        replace_sampler_ddp=False,  
        accelerator=backend,  
        seed=42,  
        deterministic=True, 
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=300, 
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback_best,checkpoint_callback_last],
        gradient_clip_val=1.0,
    )

    args,_ = parser.parse_known_args()
    args.resume_from_which_checkpoint = resume_from_which_checkpoint
    
    checkpoint_dir_general = args.default_root_dir / "lightning_logs"
    if checkpoint_dir_general.exists():
        args.resume_from_checkpoint = get_checkpoint(checkpoint_dir_general, args.resume_from_which_checkpoint)
        print('Resume from checkpoint {}'.format(args.resume_from_checkpoint))
    
    # Set arguments specific for this experiment
    dargs = vars(args)
    for key in hp.keys():
        dargs[key] = hp[key][0]
    args.seed = int(42 + 10*rr)
    
    return args

def create_fastmri_dirs_yaml(brain_path,exp_name):
    if os.path.isfile("fastmri_dirs.yaml"):
        os.remove("fastmri_dirs.yaml")
        
    with open("fastmri_dirs.yaml", "a") as text_file:
        brain_path = brain_path
        print(brain_path, file=text_file) 
        log_path = "log_path: ./{}/log_files".format(exp_name)
        print(log_path, file=text_file)     

def evaluate_reconstructions(test_mode,metrics_filename):
    path_config = pathlib.Path("fastmri_dirs.yaml")
    test_path = (fetch_dir("brain_path", path_config) / "multicoil_val") 
    predictions_path = (fetch_dir("log_path", path_config) / "reconstructions")

    if test_mode == "test_on_test":
        with open("./training_set_lists/test_filenames.yaml", 'r') as stream:
            test_filenames = yaml.safe_load(stream)
    elif test_mode == "test_on_val":
        with open("./training_set_lists/val_filenames.yaml", 'r') as stream:
            test_filenames = yaml.safe_load(stream)

    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--target-path",
        type=pathlib.Path,
        default = test_path,
        help="Path to the ground truth data",
    )
    parser.add_argument(
        "--predictions-path",
        type=pathlib.Path,
        default = predictions_path,
        help="Path to reconstructions",
    )
    parser.add_argument(
        "--challenge",
        choices=["singlecoil", "multicoil"],
        default = "multicoil",
        help="Which challenge",
    )
    parser.add_argument("--acceleration", type=int, default=None)
    parser.add_argument(
        "--acquisition",
        choices=[
            "CORPD_FBK",
            "CORPDFS_FBK",
            "AXT1",
            "AXT1PRE",
            "AXT1POST",
            "AXT2",
            "AXFLAIR",
        ],
        default=None,
        help="If set, only volumes of the specified acquisition type are used "
        "for evaluation. By default, all volumes are included.",
    )
    args,_ = parser.parse_known_args()

    recons_key = (
        "reconstruction_rss" if args.challenge == "multicoil" else "reconstruction_esc"
    )
    metrics = evaluate(args, recons_key, test_filenames)
    print(metrics)
    with open(metrics_filename, 'wb') as output:
        pickle.dump(metrics, output)
        
    
        
def get_checkpoint(checkpoint_dir_general, resume_from_which_checkpoint="last"):
    """
    Args:
        checkpoint_dir_general: path to different versions of current model
        resume_from_which_checkpoint: "last" loads the checkpoint that was added last to the latest version. 
            "best loads the best checkpoint across all versions in terms of minimal validation loss"
    Output:
        resume_from_checkpoint: path to checkpoint
    """
    versions_list = sorted(checkpoint_dir_general.glob("version_*"), key=os.path.getmtime)
    num_versions = len(versions_list)-1
    
    if resume_from_which_checkpoint=="last": 
        if versions_list:
            ckpt_list = []
            for version in versions_list:
                checkpoint_dir = version / "checkpoints"
                [ckpt_list.append(item) for item in sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)]

            last_epoch = -1  
            for ckpt in ckpt_list:
                if 'last' in str(ckpt): 
                    ind1 = str(ckpt).find('epoch=')
                    ind2 = str(ckpt).find('-step')
                    epoch = int(str(ckpt)[ind1+len('epoch='):ind2])
                    if epoch>last_epoch:
                        last_epoch = epoch
                        resume_from_checkpoint = str(ckpt)
                
    elif resume_from_which_checkpoint=="best":
        if versions_list:
            ckpt_list = []
            for version in versions_list:
                checkpoint_dir = version / "checkpoints"
                [ckpt_list.append(item) for item in sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)]

            best_val = 1.0    
            for ckpt in ckpt_list:
                ind1 = str(ckpt).find('val_loss=')
                ind2 = str(ckpt).find('.ckpt')
                if 'best' in str(ckpt) and float(str(ckpt)[ind1+len('val_loss='):ind2])<best_val:
                    best_val = float(str(ckpt)[ind1+len('val_loss='):ind2])
                    resume_from_checkpoint = str(ckpt)
                
    return resume_from_checkpoint