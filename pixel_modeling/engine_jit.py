import math
import sys
import os
import shutil

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity

from torch_fidelity.metric_fid import fid_featuresdict_to_statistics, fid_statistics_to_metric
from torch_fidelity.utils import create_feature_extractor, extract_featuresdict_from_input_id_cached

def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, wandb_run=None, args=None):
    model.train(True)
    base_model = misc.get_base_model(model_without_ddp)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    max_steps = getattr(args, "max_train_steps", None)
    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        if getattr(base_model, "ema_params1", None) is not None:
            model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if wandb_run is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                wandb_run.log(
                    {
                        'train_loss': loss_value_reduce,
                        'lr': lr,
                        'epoch': epoch,
                    },
                    step=epoch_1000x,
                )

        if max_steps is not None and (data_iter_step + 1) >= max_steps:
            break


def evaluate(model_without_ddp, sampler, args, epoch, batch_size=64, wandb_run=None):

    model_without_ddp.eval()
    base_model = misc.get_base_model(model_without_ddp)
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            sampler.method, sampler.steps, sampler.cfg_scale,
            sampler.cfg_interval[0], sampler.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # switch to ema params, hard-coded to be the first one
    swapped_to_ema = False
    current_state = None
    if getattr(base_model, "ema_params1", None) is not None:
        print("Switch to ema")
        current_state = misc.get_model_state_dict(model_without_ddp, rank0_only=False, offload_to_cpu=True)
        ema_state = misc.build_state_dict_from_ema(model_without_ddp, base_model.ema_params1)
        misc.load_model_state_dict(model_without_ddp, ema_state)
        swapped_to_ema = True

    # ensure that the number of images per class is equal.
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = sampler.generate(labels_gen)

        torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()

    # back to no ema
    if swapped_to_ema:
        print("Switch back from ema")
        misc.load_model_state_dict(model_without_ddp, current_state)

    # compute FID and IS
    if wandb_run is not None:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError

        fid = None
        inception_score = None

        try:
            # Newer torch-fidelity (with fid_statistics_file support)
            metrics_dict = torch_fidelity.calculate_metrics(
                input1=save_folder,
                input2=None,
                fid_statistics_file=fid_statistics_file,
                cuda=True,
                isc=True,
                fid=True,
                kid=False,
                prc=False,
                verbose=False,
            )
            fid = metrics_dict['frechet_inception_distance']
            inception_score = metrics_dict['inception_score_mean']
        except Exception as e:
            # Compatibility path for torch-fidelity versions that require input2 for FID
            try:
                metrics_dict_is = torch_fidelity.calculate_metrics(
                    input1=save_folder,
                    input2=None,
                    cuda=True,
                    isc=True,
                    fid=False,
                    kid=False,
                    prc=False,
                    verbose=False,
                )
                inception_score = metrics_dict_is['inception_score_mean']
            except Exception as is_err:
                print("Failed to compute IS via torch-fidelity:", is_err)

            try:
                # Compute FID manually using the pre-computed statistics file.
                feat_extractor = create_feature_extractor(
                    'inception-v3-compat',
                    ['2048'],
                    input1=save_folder,
                    input2=None,
                    cuda=True,
                    verbose=False,
                )
                featuresdict = extract_featuresdict_from_input_id_cached(
                    1, feat_extractor, input1=save_folder, input2=None, cuda=True, verbose=False
                )
                fid_stat_gen = fid_featuresdict_to_statistics(featuresdict, '2048')
                fid_stat_ref_npz = np.load(fid_statistics_file)
                fid_stat_ref = {'mu': fid_stat_ref_npz['mu'], 'sigma': fid_stat_ref_npz['sigma']}
                fid = fid_statistics_to_metric(fid_stat_gen, fid_stat_ref, verbose=False)['frechet_inception_distance']
            except Exception as fid_err:
                print("Failed to compute FID with reference stats:", fid_err)

        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_payload = {'epoch': epoch}
        if fid is not None:
            log_payload['fid{}'.format(postfix)] = fid
        if inception_score is not None:
            log_payload['is{}'.format(postfix)] = inception_score
        if len(log_payload) > 1:  # ensure at least one metric is present
            wandb_run.log(log_payload, step=epoch)
        if fid is not None and inception_score is not None:
            print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        elif fid is not None:
            print("FID: {:.4f}".format(fid))
        elif inception_score is not None:
            print("Inception Score: {:.4f}".format(inception_score))
        shutil.rmtree(save_folder)

    torch.distributed.barrier()
