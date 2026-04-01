import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
from functools import partial

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc

import copy
from engine_jit import train_one_epoch, evaluate

from denoiser import Denoiser
from model_jit import JiTBlock
from sampling import DiffusionSampler


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--prediction', default='x', type=str,
                        choices=['x', 'v', 'eps', 'epsilon'],
                        help='Model output type (x, v, eps); loss always uses velocity')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--window_size', default=None, type=int,
                        help='Optional DCT window size for dynamic patchify; defaults to patch_size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--eval_at_start', action='store_true',
                        help='Run evaluation once before training starts (useful for quick smoke tests)')
    parser.add_argument('--disable_eval', action='store_true',
                        help='Force-skip all eval hooks during training (overrides online_eval / eval_at_start)')
    parser.add_argument('--max_train_steps', type=int, default=None,
                        help='Optional cap on training steps per epoch for quick debugging')
    parser.add_argument('--eval_num_images', type=int, default=None,
                        help='Override num_images just for eval (must be divisible by class_num); helpful for quick tests')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1000, type=int)

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--disable_checkpoint', action='store_true',
                        help='Skip all checkpoint saving (useful when debugging hangs on slow filesystems)')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')
    parser.add_argument('--use_fsdp', action='store_true',
                        help='Enable FSDP training instead of DDP')
    parser.add_argument('--fsdp_sharding', default='full', choices=['full', 'shard_grad'],
                        help='FSDP sharding strategy')
    parser.add_argument('--fsdp_precision', default='bf16', choices=['bf16', 'fp16', 'fp32'],
                        help='Mixed precision policy for FSDP parameters/reductions')
    parser.add_argument('--fsdp_ckpt_type', default='full', choices=['full', 'local'],
                        help='FSDP checkpoint type: full (all-gather to rank0) or local (per-rank shards to avoid all-gather)')
    parser.add_argument('--disable_ema', action='store_true',
                        help='Disable EMA tracking (enabled by default)')
    parser.add_argument('--wandb_project', default='jit', type=str,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='Weights & Biases entity/user (optional)')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='Optional Weights & Biases run name')
    parser.add_argument('--wandb_mode', default='online', type=str, choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases mode')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    if not hasattr(args, 'gpu'):
        args.gpu = 0

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up Weights & Biases logging (only on main process)
    wandb_run = None
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.wandb_mode != 'disabled':
            import wandb  # lazy import to avoid hooking stdout when disabled
            run_name = args.wandb_run_name or f"{args.model}-seed{args.seed}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
                dir=args.output_dir,
                mode=args.wandb_mode,
                resume="allow"
            )

    # Data augmentation transforms
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor()
    ])

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)


    if args.use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy  # type: ignore
        from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy  # type: ignore

        mp_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[args.fsdp_precision]
        mp_policy = MixedPrecision(
            param_dtype=mp_dtype,
            reduce_dtype=mp_dtype,
            buffer_dtype=mp_dtype,
        )
        sharding = ShardingStrategy.FULL_SHARD if args.fsdp_sharding == 'full' else ShardingStrategy.SHARD_GRAD_OP
        auto_wrap_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda m: isinstance(m, JiTBlock))
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=args.gpu,
            sharding_strategy=sharding,
            mixed_precision=mp_policy,
            sync_module_states=True,
            use_orig_params=True,
            limit_all_gathers=True,
        )
        model_without_ddp = model
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        else:
            model_without_ddp = model

    sampler = DiffusionSampler(
        model_without_ddp,
        img_size=args.img_size,
        num_classes=args.class_num,
        noise_scale=args.noise_scale,
        t_eps=args.t_eps,
        method=args.sampling_method,
        steps=args.num_sampling_steps,
        cfg_scale=args.cfg,
        cfg_interval=(args.interval_min, args.interval_max),
    )

    base_model = misc.get_base_model(model_without_ddp)

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        misc.load_model_state_dict(model_without_ddp, checkpoint.get('model'))

        if not args.disable_ema:
            base_model.ema_params1 = misc.extract_ema_params_from_state(model_without_ddp, checkpoint.get('model_ema1'))
            base_model.ema_params2 = misc.extract_ema_params_from_state(model_without_ddp, checkpoint.get('model_ema2'))
        else:
            base_model.ema_params1 = None
            base_model.ema_params2 = None
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            misc.load_optimizer_state_dict(model_without_ddp, optimizer, checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        if args.disable_ema:
            base_model.ema_params1 = None
            base_model.ema_params2 = None
        else:
            base_model.ema_params1 = {name: p.detach().clone() for name, p in base_model.named_parameters()}
            base_model.ema_params2 = {name: p.detach().clone() for name, p in base_model.named_parameters()}
        print("Training from scratch")

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        if args.eval_num_images is not None:
            args = copy.deepcopy(args)
            args.num_images = args.eval_num_images
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, sampler, args, 0, batch_size=args.gen_bsz, wandb_run=wandb_run)
        return

    # Training loop
    if args.online_eval and args.eval_at_start and not args.disable_eval:
        eval_args = args
        if args.eval_num_images is not None:
            eval_args = copy.deepcopy(args)
            eval_args.num_images = args.eval_num_images
        torch.cuda.empty_cache()
        with torch.no_grad():
            print("Running evaluation before training starts (eval_at_start)")
            evaluate(model_without_ddp, sampler, eval_args, args.start_epoch, batch_size=args.gen_bsz, wandb_run=wandb_run)
        torch.cuda.empty_cache()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, wandb_run=wandb_run, args=args)

        # Save checkpoint periodically (unless explicitly disabled)
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )
        if epoch % 100 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and not args.disable_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, sampler, args, epoch, batch_size=args.gen_bsz, wandb_run=wandb_run)
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)
    if misc.is_main_process() and wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
