# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import datetime
import io
import logging
import logging.handlers
import os
import sys
import time
from collections import defaultdict, deque
from contextlib import ContextDecorator
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
import wandb
from torch.utils.tensorboard import SummaryWriter

import local_utils.distributed as distributed
from local_utils.distributed import is_dist_avail_and_initialized

_logger = logging.getLogger('logger')


def init_wandb_logger(args, output_dir):

    wandb_id = None
    if distributed.is_main_process():
        # TODO: reactivate wandb
        # if os.path.exists(os.path.join(output_dir, "wandb_id.txt")) and not args.cineca:
        #     with open(os.path.join(output_dir, "wandb_id.txt"), "r") as f:
        #         wandb_id = f.read()
        #         _logger.info(f"\t-Found existing wandb_id: {wandb_id}" + "\n")
        #else:
        wandb_id = wandb.util.generate_id()
        _logger.info(f"\t-Creating new wandb_id: {wandb_id}" + "\n")
        with open(os.path.join(output_dir, "wandb_id.txt"), "w") as f:
            f.write(wandb_id)
    args.wandb_id = wandb_id

    if not args.wandb_name:
        args.wandb_name = output_dir.name

    _logger.info(f">>> Creating wandb logger with ID: `{wandb_id}`")
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        config=args,
        name=args.wandb_name,
        resume=args.wandb_name,
        id=wandb_id,
        group=args.wandb_group,
        notes=args.wandb_notes,
    )

    wandb.define_metric("batch_step_train")  # define which metrics will be plotted against it
    wandb.define_metric("batch_step_val")  # define which metrics will be plotted against it
    wandb.define_metric("loss_training", step_metric="batch_step_train")
    wandb.define_metric("loss_validation", step_metric="batch_step_val")


class TensorboardWriter:
    def __init__(self, log_dir, enabled):
        self.writer = SummaryWriter(log_dir) if enabled else None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

        self.step = 0

        self.tb_writer_ftns = {
            "add_figure",
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding",
            # "add_graph",
            # "add_hparams",
        }

    def set_step(self, step):
        self.step = step

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:
            attr = getattr(self.writer, name)
            return attr


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
            value=self.value
        )


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.eps = 1e-8

    def update(self, a, b):
        n = self.num_classes

        if self.mat.device != a.device:
            self.mat = self.mat.to(a.device)

        with torch.no_grad():
            k = (a >= 0) & (a < n)  # exclude ignored indices (i.e., the 'ignore_index' in the loss)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / (h.sum() + self.eps)
        acc = torch.diag(h) / (h.sum(1) + self.eps)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + self.eps)
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            '\t-Global Accuracy: {:.3f}\n'
            '\t-mean Accuracy: {:.3f}\n'
            '\t-Accuracy per class: {}\n'
            '\t-IoU: {}\n'
            '\t-mean IoU: {:.3f}').format(
            acc_global.item() * 100,
            acc.mean().item() * 100,
            ['{:.3f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.3f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)  # defaultdict assigns "SmoothValue" by default to not existing keys
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                type(self).__name__, attr
            )
        )

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
            '[{0}] -',
            header,
            '[{1' + space_fmt + '}/{2}]',
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
            yield i, obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            datetime.datetime.now(),
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            datetime.datetime.now(),
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            '{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def save_checkpoint(
        args,
        epoch,
        output_dir,
        model_without_ddp,
        optimizer,
        lr_scheduler,
        loss_scaler,
        test_stats,
        best_metric,
        best_epoch,
        best_accuracy,
        wandb_logger,
        model_ema=None,
):
    checkpoint_paths = [output_dir / f'checkpoint-latest.pth']

    if args.save_all:
        checkpoint_paths.append(output_dir / f'checkpoint-epoch-{epoch}.pth')

    if (test_stats[best_metric] > best_accuracy and best_metric != "loss") or (
            test_stats[best_metric] < best_accuracy and best_metric == "loss"):
        best_accuracy = test_stats[best_metric]
        best_epoch = epoch

        for k, v in test_stats.items():
            wandb_logger.write_to_summary(f"best_{k}", v)
        wandb_logger.write_to_summary("best_epoch", best_epoch)

        checkpoint_paths.append(output_dir / 'best.pth')

    checkpoint_state_dict = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'best_accuracy': best_accuracy,
        'best_epoch': best_epoch,
        # 'model_ema': get_state_dict(model_ema),
        'args': args,
    }
    if loss_scaler:
        checkpoint_state_dict['scaler'] = loss_scaler.state_dict()

    for checkpoint_path in checkpoint_paths:
        distributed.save_on_master(checkpoint_state_dict, checkpoint_path)

    return best_accuracy, best_epoch


class measure_elapsed_time(ContextDecorator):
    def __init__(self, label=None, active=True):
        self.label = label
        self.active = active and distributed.is_main_process()

    def __call__(self, func):
        if self.label is None:
            self.label = func.__name__
        return super().__call__(func)

    def __enter__(self):
        if self.active:
            self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        if self.active:
            self.time = perf_counter() - self.time
            _logger = logging.getLogger()
            _logger.setLevel(logging.INFO)
            _logger.info(f'{self.label} took {self.time:.5f} seconds')


# define logging format and steam handler
class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)


def custom_setup_default_logging(default_level=logging.INFO, log_path=''):
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('[%(asctime)s] - %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    console_handler.setFormatter(console_formatter)  # FormatterNoInfo()
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter("%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)
