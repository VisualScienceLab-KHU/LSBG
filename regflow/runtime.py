import os
import random
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self):
        return self.rank == 0

    def print(self, *args, **kwargs):
        if self.is_main:
            print(*args, **kwargs, flush=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@contextmanager
def initialize(config):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(config["runtime"]["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Install the CUDA build of PyTorch and allocate a GPU."
            )
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} GPUs are visible."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=30)
        )
    seed = config["runtime"]["seed"] + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    runtime = Runtime(device, rank, world_size, local_rank)
    try:
        yield runtime
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn_worker(rank, worker, config, port):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(config["runtime"]["num_processes"]),
    )
    with initialize(config) as runtime:
        worker(config, runtime)


def launch(worker, config):
    count = config["runtime"]["num_processes"]
    external = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if external:
        with initialize(config) as runtime:
            runtime.print(
                f"External launcher: {runtime.world_size} process(es); runtime.num_processes is ignored."
            )
            worker(config, runtime)
        return
    if config["runtime"]["device"] == "cuda" and torch.cuda.device_count() < count:
        raise RuntimeError(
            f"Requested {count} GPU processes; {torch.cuda.device_count()} GPUs are visible. "
            "Set runtime.num_processes to your allocated GPU count."
        )
    if count == 1:
        with initialize(config) as runtime:
            worker(config, runtime)
        return
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_spawn_worker, args=(worker, config, port), nprocs=count, join=True)


def all_reduce_sum(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def check_finite(loss):
    finite = torch.isfinite(loss.detach()).to(torch.int32)
    if dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError(
            "Non-finite loss on at least one process; no optimizer step was applied."
        )
