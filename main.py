import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

from taming.data.utils import custom_collate


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, 
                 batch_size, 
                 train=None, 
                 validation=None, 
                 test=None,
                 wrap=False, 
                 num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        
        # Improved num_workers calculation
        if num_workers is None:
            try:
                # Use min of batch_size*2 and available CPU cores
                self.num_workers = min(batch_size*2, os.cpu_count() or 2)
            except:
                self.num_workers = 2
        else:
            self.num_workers = num_workers
        
        # Dataset configuration
        self.dataset_configs = {
            "train": train,
            "validation": validation,
            "test": test
        }
        self.wrap = wrap
        
        # Datasets will be populated in setup
        self.datasets = {}

    def prepare_data(self):
        """Prepare data - called only on 1 GPU in distributed settings"""
        for name, data_cfg in self.dataset_configs.items():
            if data_cfg is not None:
                instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        """Set up datasets for each stage"""
        self.datasets = {}
        for name, data_cfg in self.dataset_configs.items():
            if data_cfg is not None:
                dataset = instantiate_from_config(data_cfg)
                
                # Optional wrapping
                if self.wrap:
                    dataset = WrappedDataset(dataset)
                
                self.datasets[name] = dataset

    def _create_dataloader(self, dataset, shuffle=False):
        """Common dataloader creation method"""
        return DataLoader(
            dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=custom_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )

    def train_dataloader(self):
        """Create train dataloader"""
        if "train" not in self.datasets:
            raise ValueError("Train dataset not configured")
        return self._create_dataloader(self.datasets["train"], shuffle=True)

    def val_dataloader(self):
        """Create validation dataloader"""
        if "validation" not in self.datasets:
            raise ValueError("Validation dataset not configured")
        return self._create_dataloader(self.datasets["validation"], shuffle=False)

    def test_dataloader(self):
        """Create test dataloader"""
        if "test" not in self.datasets:
            raise ValueError("Test dataset not configured")
        return self._create_dataloader(self.datasets["test"], shuffle=False)

    def predict_dataloader(self):
        """Optional predict dataloader"""
        # You can implement this if needed
        if "test" in self.datasets:
            return self._create_dataloader(self.datasets["test"], shuffle=False)
        return None

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            # Use OmegaConf.to_yaml() instead of .pretty()
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            # Use OmegaConf.to_yaml() instead of .pretty()
            print(OmegaConf.to_yaml(OmegaConf.create({"lightning": self.lightning_config})))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

class ImageLogger(Callback):
    def __init__(self, 
                 batch_frequency=2000, 
                 max_images=4, 
                 clamp=True, 
                 increase_log_steps=True,
                 log_all_val=False,
                 disabled=False):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_all_val = log_all_val
        self.disabled = disabled
        
        # Logger mapping for different logger types
        self.logger_log_images = {
            pl.loggers.TensorBoardLogger: self._tensorboard_logging,
        }
        
        # Calculate log steps
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        
        self.clamp = clamp

    @rank_zero_only
    def _tensorboard_logging(self, pl_module, images, batch_idx, split):
        """Log images to TensorBoard"""
        try:
            for k, img_tensor in images.items():
                # Ensure image is in correct range
                grid = torchvision.utils.make_grid(img_tensor)
                grid = (grid + 1.0) / 2.0  # Normalize to 0-1 range
                
                tag = f"{split}/{k}"
                pl_module.logger.experiment.add_image(
                    tag, grid,
                    global_step=pl_module.global_step
                )
        except Exception as e:
            print(f"Error in TensorBoard logging: {e}")

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        """Save images locally"""
        try:
            root = os.path.join(save_dir, "images", split)
            os.makedirs(root, exist_ok=True)

            for k, img_tensor in images.items():
                # Create grid
                grid = torchvision.utils.make_grid(img_tensor, nrow=4)
                
                # Normalize and convert to numpy
                grid = (grid + 1.0) / 2.0  # Normalize to 0-1
                grid = grid.permute(1, 2, 0).cpu().numpy()
                grid = (grid * 255).astype(np.uint8)

                # Create filename
                filename = f"{k}_gs-{global_step:06d}_e-{current_epoch:06d}_b-{batch_idx:06d}.png"
                path = os.path.join(root, filename)

                # Save image
                Image.fromarray(grid).save(path)
        except Exception as e:
            print(f"Error saving local images: {e}")

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        """Main image logging method"""
        try:
            # Check if logging is needed
            if not self.check_frequency(batch_idx):
                return

            # Ensure log_images method exists
            if not hasattr(pl_module, "log_images") or not callable(pl_module.log_images):
                print("log_images method not found in the module")
                return

            # Temporarily switch to eval mode
            was_training = pl_module.training
            pl_module.eval()

            # Disable gradient computation
            with torch.no_grad():
                # Get images to log
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            # Process images
            processed_images = {}
            for k, img_tensor in images.items():
                # Limit number of images
                N = min(img_tensor.shape[0], self.max_images)
                img_tensor = img_tensor[:N]

                # Ensure tensor and clamp if needed
                if isinstance(img_tensor, torch.Tensor):
                    img_tensor = img_tensor.detach().cpu()
                    if self.clamp:
                        img_tensor = torch.clamp(img_tensor, -1., 1.)
                
                processed_images[k] = img_tensor

            # Log images
            if processed_images:
                # Local file logging
                self.log_local(
                    pl_module.logger.save_dir, 
                    split, 
                    processed_images,
                    pl_module.global_step, 
                    pl_module.current_epoch, 
                    batch_idx
                )

                # Logger-specific logging
                logger_type = type(pl_module.logger)
                logger_log_images = self.logger_log_images.get(logger_type, lambda *args, **kwargs: None)
                logger_log_images(pl_module, processed_images, batch_idx, split)

            # Restore original training state
            if was_training:
                pl_module.train()

        except Exception as e:
            print(f"Error in image logging: {e}")
            # Restore training state in case of error
            if was_training:
                pl_module.train()

    def check_frequency(self, batch_idx):
        """Determine if logging should occur"""
        if self.disabled:
            return False
        
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        """Callback for end of training batch"""
        try:
            self.log_img(pl_module, batch, batch_idx, split="train")
        except Exception as e:
            print(f"Error in on_train_batch_end: {e}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        """Callback for end of validation batch"""
        try:
            if self.log_all_val:
                self.log_img(pl_module, batch, batch_idx, split="val")
        except Exception as e:
            print(f"Error in on_validation_batch_end: {e}")




if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix
        logdir = os.path.join("logs", nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["distributed_backend"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        # NOTE wandb < 0.10.0 interferes with shutdown
        # wandb >= 0.10.0 seems to fix it but still interferes with pudb
        # debugging (wrongly sized pudb ui)
        # thus prefer testtube for now
        default_logger_cfgs = {
    "tensorboard": {
        "target": "pytorch_lightning.loggers.tensorboard.TensorBoardLogger",
        "params": {
            "name": nowname,
            "save_dir": logdir,
            "default_hp_metric": False,
        }
    }
        }

        default_logger_cfg = default_logger_cfgs["tensorboard"]
        custom_logger_cfg = OmegaConf.select(lightning_config, "logger", default=OmegaConf.create())
        logger_cfg = OmegaConf.merge(default_logger_cfg, custom_logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
    "target": "pytorch_lightning.callbacks.ModelCheckpoint",
    "params": {
        "dirpath": ckptdir,
        "filename": "{epoch:06}",
        "verbose": True,
        "save_last": True,
    }
}
        if hasattr(model, "monitor"):
          print(f"Monitoring {model.monitor} as checkpoint metric.")
          default_modelckpt_cfg["params"]["monitor"] = model.monitor
          default_modelckpt_cfg["params"]["save_top_k"] = 3

        modelckpt_cfg = OmegaConf.create(lightning_config.get('modelcheckpoint', {}))
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        modelckpt_callback = instantiate_from_config(modelckpt_cfg)

        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 500,
                    "max_images": 4,
                    "clamp": True,
                    
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    #"log_momentum": True
                }
            },
        }
        if "checkpoint_callback" in trainer_kwargs:
            del trainer_kwargs["checkpoint_callback"]
        callbacks_cfg = OmegaConf.create(lightning_config.get('callbacks', {}))
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

        # Initialize callbacks list
        trainer_kwargs["callbacks"] = []

        # Add ModelCheckpoint callback
        trainer_kwargs["callbacks"].append(modelckpt_callback)

        # Add other callbacks from configuration
        for k in callbacks_cfg:
            callback = instantiate_from_config(callbacks_cfg[k])
            trainer_kwargs["callbacks"].append(callback)


        trainer_kwargs['max_epochs'] = 1000
        trainer_kwargs['log_every_n_steps'] = 5
        trainer_kwargs['gradient_clip_val'] = 0.5

        # Add precision and accelerator settings
        if not cpu:  # Only if not running on CPU
              trainer_kwargs['precision'] = 16  # Mixed precision
              trainer_kwargs['accelerator'] = 'gpu'
              trainer_kwargs['devices'] = gpuinfo if isinstance(gpuinfo, int) else len(gpuinfo)

        # Optionally add learning rate scheduling callback
        lr_monitor = LearningRateMonitor(logging_interval='step')
        if lr_monitor not in trainer_kwargs.get('callbacks', []):
              trainer_kwargs['callbacks'].append(lr_monitor)

        # Create trainer
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()

        # configure learning rate
        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
          ngpu = 1

        # Safely retrieve accumulate_grad_batches with a default value of 1
        accumulate_grad_batches = lightning_config.trainer.get('accumulate_grad_batches', 1)
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")

        # Ensure the value is set in the trainer configuration
        lightning_config.trainer['accumulate_grad_batches'] = accumulate_grad_batches

        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
