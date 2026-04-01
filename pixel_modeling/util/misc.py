import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from typing import Optional, Dict, Any


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            try:
                builtin_print('[{}] '.format(now), end='')  # print with time stamp
                builtin_print(*args, **kwargs)
            except BrokenPipeError:
                # Ignore broken pipe to avoid crashing when stdout is closed.
                return

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def add_weight_decay(model, weight_decay=0, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'diffloss' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def save_model(args, model_without_ddp, optimizer, epoch, epoch_name=None):
    if epoch_name is None:
        epoch_name = str(epoch)
    output_dir = Path(args.output_dir)
    rank = get_rank()
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % epoch_name)
    ckpt_type = getattr(args, "fsdp_ckpt_type", "full")

    # Fast path: local FSDP shards saved per-rank to avoid all-gather
    if ckpt_type == "local" and is_fsdp_model(model_without_ddp):
        local_model_state = get_model_state_dict(
            model_without_ddp, rank0_only=False, offload_to_cpu=True, state_dict_type="local"
        )
        local_optim_state = get_optimizer_state_dict(
            model_without_ddp, optimizer, rank0_only=False, offload_to_cpu=True, state_dict_type="local"
        )
        base_model = get_base_model(model_without_ddp)
        ema_state1 = build_state_dict_from_ema(model_without_ddp, getattr(base_model, "ema_params1", None))
        ema_state2 = build_state_dict_from_ema(model_without_ddp, getattr(base_model, "ema_params2", None))

        local_path = output_dir / (f'checkpoint-{epoch_name}-rank{rank}.pth')
        to_save = {
            'model': local_model_state,
            'optimizer': local_optim_state,
            'epoch': epoch,
            'args': args,
        }
        if ema_state1 is not None:
            to_save['model_ema1'] = ema_state1
        if ema_state2 is not None:
            to_save['model_ema2'] = ema_state2
        torch.save(to_save, local_path)
        torch.distributed.barrier()
        return

    model_state = get_model_state_dict(model_without_ddp, rank0_only=True, offload_to_cpu=True, state_dict_type="full")
    optim_state = get_optimizer_state_dict(model_without_ddp, optimizer, rank0_only=True, offload_to_cpu=True, state_dict_type="full")
    base_model = get_base_model(model_without_ddp)

    to_save = {
        'model': model_state,
        'optimizer': optim_state,
        'epoch': epoch,
        'args': args,
    }

    # Save EMA weights by building state dicts from the stored EMA params.
    ema_state1 = build_state_dict_from_ema(model_without_ddp, getattr(base_model, "ema_params1", None))
    ema_state2 = build_state_dict_from_ema(model_without_ddp, getattr(base_model, "ema_params2", None))
    if ema_state1 is not None:
        to_save['model_ema1'] = ema_state1
    if ema_state2 is not None:
        to_save['model_ema2'] = ema_state2

    save_on_master(to_save, checkpoint_path)


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def is_fsdp_model(model: torch.nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel  # type: ignore
    except Exception:
        return False
    return isinstance(model, FullyShardedDataParallel)


def get_model_state_dict(model: torch.nn.Module, rank0_only: bool = True, offload_to_cpu: bool = True,
                         state_dict_type: str = "full") -> Optional[Dict[str, Any]]:
    if rank0_only and (not is_main_process()):
        return None
    if is_fsdp_model(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig, LocalStateDictConfig  # type: ignore
        if state_dict_type == "local":
            # Older PyTorch LocalStateDictConfig may not accept rank0_only; local shards are always per-rank.
            try:
                state_dict_cfg = LocalStateDictConfig(rank0_only=rank0_only, offload_to_cpu=offload_to_cpu)
            except TypeError:
                state_dict_cfg = LocalStateDictConfig(offload_to_cpu=offload_to_cpu)
            sd_type = StateDictType.LOCAL_STATE_DICT
        else:
            state_dict_cfg = FullStateDictConfig(rank0_only=rank0_only, offload_to_cpu=offload_to_cpu)
            sd_type = StateDictType.FULL_STATE_DICT
        with FSDP.state_dict_type(model, sd_type, state_dict_cfg):
            return model.state_dict()
    state = model.state_dict()
    if offload_to_cpu:
        state = {k: v.cpu() for k, v in state.items()}
    return state


def load_model_state_dict(model: torch.nn.Module, state_dict: Optional[Dict[str, Any]]):
    if state_dict is None:
        return
    if is_fsdp_model(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig  # type: ignore
        state_dict_cfg = FullStateDictConfig(rank0_only=False, offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_dict_cfg):
            model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def get_optimizer_state_dict(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
                             rank0_only: bool = True, offload_to_cpu: bool = True,
                             state_dict_type: str = "full") -> Optional[Dict[str, Any]]:
    if optimizer is None:
        return None
    if rank0_only and not is_main_process():
        return None
    if is_fsdp_model(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig, FullOptimStateDictConfig, LocalStateDictConfig, LocalOptimStateDictConfig  # type: ignore
        if state_dict_type == "local":
            try:
                state_dict_cfg = LocalStateDictConfig(rank0_only=rank0_only, offload_to_cpu=offload_to_cpu)
            except TypeError:
                state_dict_cfg = LocalStateDictConfig(offload_to_cpu=offload_to_cpu)
            try:
                optim_state_cfg = LocalOptimStateDictConfig(rank0_only=rank0_only, offload_to_cpu=offload_to_cpu)
            except TypeError:
                optim_state_cfg = LocalOptimStateDictConfig(offload_to_cpu=offload_to_cpu)
            sd_type = StateDictType.LOCAL_STATE_DICT
        else:
            state_dict_cfg = FullStateDictConfig(rank0_only=rank0_only, offload_to_cpu=offload_to_cpu)
            optim_state_cfg = FullOptimStateDictConfig(rank0_only=rank0_only, offload_to_cpu=offload_to_cpu)
            sd_type = StateDictType.FULL_STATE_DICT
        with FSDP.state_dict_type(model, sd_type, state_dict_cfg):
            try:
                return FSDP.optim_state_dict(model, optimizer, optim_state_dict_config=optim_state_cfg)
            except TypeError:
                # Older PyTorch versions do not accept optim_state_dict_config.
                return FSDP.optim_state_dict(model, optimizer)
    return optimizer.state_dict()


def load_optimizer_state_dict(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], optim_state_dict: Optional[Dict[str, Any]]):
    if optimizer is None or optim_state_dict is None:
        return
    if is_fsdp_model(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
        try:
            optim_state = FSDP.optim_state_dict_to_load(optim_state_dict, model)
            optimizer.load_state_dict(optim_state)
        except Exception:
            # Fallback to direct load if helper API differs.
            optimizer.load_state_dict(optim_state_dict)
    else:
        optimizer.load_state_dict(optim_state_dict)


def swap_params(model: torch.nn.Module, params: Optional[list], param_list: Optional[list] = None):
    if params is None:
        return
    if param_list is None:
        param_list = list(model.parameters())
    for p, q in zip(param_list, params):
        p.data, q.data = q.data, p.data


def build_state_dict_from_ema(model: torch.nn.Module, ema_params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if ema_params is None:
        return None
    state = get_model_state_dict(model, rank0_only=True)
    if state is None:
        return None
    for name, value in ema_params.items():
        if name in state:
            state[name] = value.detach().cpu()
    return state


def extract_ema_params_from_state(model: torch.nn.Module, ema_state_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if ema_state_dict is None:
        return None
    param_names = {name for name, _ in model.named_parameters()}
    ema_params = {k: v.detach().clone() for k, v in ema_state_dict.items() if k in param_names}
    return ema_params


def get_base_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the underlying module for wrappers (DDP/FSDP) so we can attach attributes like EMA.
    """
    if hasattr(model, "_fsdp_wrapped_module"):
        return model._fsdp_wrapped_module  # type: ignore[attr-defined]
    if hasattr(model, "module"):
        return model.module  # e.g., DDP
    return model
