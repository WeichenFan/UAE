import os
import sys
from omegaconf import OmegaConf
import logging
from typing import Optional, Tuple
import shutil
from .wandb_utils import initialize, create_logger
from .io_utils import is_disk_full_error
import logging

def _create_stdout_logger(logger_name: str = "uae") -> logging.Logger:
    logger = logging.getLogger(f"{logger_name}_stdout")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(
            logging.Formatter(
                '[\033[34m%(asctime)s\033[0m] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
            )
        )
        logger.addHandler(stream_handler)
    return logger

def configure_experiment_dirs(args, rank) -> Tuple[str, str, logging.Logger]:
    experiment_name = os.environ.get("EXPERIMENT_NAME")
    assert experiment_name is not None, "Please set the EXPERIMENT_NAME environment variable."
    experiment_dir = os.path.join(args.results_dir, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints") 
    if rank == 0:
        try:
            os.makedirs(args.results_dir, exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir, 'uae')
            logger.info(f"Experiment directory created at {experiment_dir}")
        except Exception as exc:
            if not is_disk_full_error(exc):
                raise
            logger = _create_stdout_logger("uae")
            logger.warning(
                "Disk appears full while preparing experiment directories under %s. "
                "Continuing without file-backed local logging/checkpoint guarantees.",
                experiment_dir,
            )
        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None, 'uae')
    return experiment_dir, checkpoint_dir, logger
def find_resume_checkpoint(resume_dir) -> Optional[str]:
    """
    Find the latest checkpoint file in the resume directory.
    Args:
        resume_dir (str): Path to the resume directory.
    Returns:
        str: Path to the latest checkpoint file.
    """
    # In multi-node launches non-zero ranks may reach this check before the
    # master rank creates/syncs the experiment directory on shared storage.
    # Missing directories should be treated as "no checkpoint yet".
    if not os.path.exists(resume_dir):
        return None
    checkpoint_dir = os.path.join(resume_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt") or f.endswith(".ckpt") or f.endswith(".safetensor")
    ]
    if len(checkpoints) == 0:
        return None
    # sort via epoch, saved as 'ep-{epoch:07d}.pt'
    checkpoints = sorted(
        checkpoints,
        key=lambda x: int(os.path.basename(x).split("-")[1].split(".")[0]),
    )
    return checkpoints[-1]

def save_worktree(
    path: str,
    config: OmegaConf,  
) -> bool:
    config_path = os.path.join(path, "config.yaml")
    src_target = os.path.join(path, "src/")
    worktree_path = os.path.join(os.getcwd(), "src")
    try:
        OmegaConf.save(config, config_path)
        shutil.copytree(worktree_path, src_target, dirs_exist_ok=True)
        print(f'Worktree {worktree_path} saved to {src_target}')
        return True
    except Exception as exc:
        if is_disk_full_error(exc):
            print(
                f"[I/O] Disk full while saving worktree/config snapshot into {path}. "
                "Skipping snapshot and continuing."
            )
            return False
        raise

if __name__ == "__main__":
    experiment_dir = sys.argv[1]
    ckpt_path = find_resume_checkpoint(experiment_dir)
    print(f"Latest checkpoint found at: {ckpt_path}")
