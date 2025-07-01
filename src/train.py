#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import logging
import os
from contextlib import suppress
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from timm import utils
from timm.layers import convert_sync_batchnorm
from timm.models import create_model, safe_model_name, load_checkpoint
from local_utils._helpers import resume_checkpoint, cosine_scheduler, extract_best_epochs_from_summary
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from visualization.embedding_visualization import visualization
from collections import OrderedDict
import torchvision.transforms as transforms

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError as e:
    has_functorch = False
from engine import train_one_epoch, validate
import local_utils.distributed as distributed
from args_train import get_args_parser, _parse_args
from local_utils.logger import init_wandb_logger, custom_setup_default_logging
from dataset.dataloader import create_loader_webdataset
from dataset.CustomTransform import MassiveTransform
from dataset.standardtransform import PipelineAugmentation
from losses.function_loss import InfoNCE
from quantitative_evaluation import classifier_training, classifier_eval, Model, Models
from local_utils.early_stopping import EarlyStopper

has_compile = hasattr(torch, 'compile')

_logger = logging.getLogger('train')


def main():
    # define the args
    parser, config_parser = get_args_parser()
    args, args_text = _parse_args(parser, config_parser)

    args.prefetcher = not args.no_prefetcher
    args.grad_accum_steps = max(1, args.grad_accum_steps)  # we fix it at 1
    distributed.init_distributed_device(args)
    device = args.device
    # all cuda setting inside the set_random_seed function
    distributed.set_random_seed(args.seed, deterministic=args.deterministic, benchmark=args.benchmark)
    multi_process_cfg = {
        "num_workers": args.workers,
        "mp_start_method": None,
        "opencv_num_threads": None,
        "omp_num_threads": None,
        "mkl_num_threads": None,
    }
    distributed.setup_multi_processes(multi_process_cfg)
    dash_line = '-' * 60 + '\n'
    if args.experiment:
        exp_name = args.experiment
    else:
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            safe_model_name(args.model),
        ])
    output_dir = Path(args.output) / args.experiment
    if args.rank == 0:
        # output_dir.mkdir(parents=True, exist_ok=True)

        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        output_dir = Path(output_dir)
        custom_setup_default_logging(log_path=os.path.join(output_dir, "log.txt"))
        env_var = sorted([(k, v) for k, v in os.environ.items()], key=(lambda x: x[0]))
        # log env variables
        env_info = '\n'.join([f'-{k}: {v}' for k, v in env_var])
        _logger.info('Environment info:\n' + env_info + '\n' + dash_line)
        # log args
        args_info = '\n'.join([f'-{k}: {v}' for k, v in sorted(vars(args).items())])
        _logger.info(dash_line + '\nArgs info:\n' + args_info + '\n' + dash_line)
        if args.log_wandb:
            _logger.info(f">>> Setup wandb: {'ON' if args.log_wandb else 'OFF'}")
            if has_wandb:
                init_wandb_logger(args, output_dir)
            else:
                _logger.warning("You've requested to log metrics to wandb but package not found. "
                                "Metrics not being logged to wandb, try `pip install wandb`")

    if not args.resume and os.path.exists(output_dir / "last.pth.tar"):
        args.resume = str(output_dir / "last.pth.tar")

    if args.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            use_amp = 'apex'
            assert args.amp_dtype == 'float16'
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            use_amp = 'native'
            assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]
    if not args.pretrained:
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            in_chans=in_chans,
            num_classes=0,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path=args.initial_checkpoint,
            **args.model_kwargs,
        )
    else:
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            **args.model_kwargs,
        )
        if 'in21k' in args.model:
            model.head = nn.Identity()
        elif args.only_validate:
            pass
        else:
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1, bias=True)
            # init on with normal truncated as per WANG et al. 2020
            torch.nn.init.normal_(model.fc.weight.data, 0.0, 0.02)

        args.mean = model.pretrained_cfg['mean']
        args.std = model.pretrained_cfg['std']
        args.crop_pct = model.pretrained_cfg['crop_pct']
        args.input_size = model.pretrained_cfg['input_size']
    if args.head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(args.head_init_scale)
            model.get_classifier().bias.mul_(args.head_init_scale)
    if args.head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, args.head_init_bias)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)
    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.rank == 0:
        if args.log_wandb:
            wandb.run.summary["#Params_total"] = n_params_total
            wandb.run.summary["#Params_requires_grad"] = n_params_grad
        _logger.info(
            f'Model {safe_model_name(args.model)} created:'
            f'\n\t-#Params total:         {n_params_total}'
            f'\n\t-#Params requires_grad: {n_params_grad}'
        )
    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    # enable split bn (separate bn stats per batch-portion)

    # move model to GPU, enable channels last layout if set
    model.to(device=device)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:  # TODO: control if we need
        args.dist_bn = ''  # disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not args.torchcompile
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
                f'and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.')

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )
    if args.weight_decay_end is not None:
        weight_decay_schedule = cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs)
    # take optimizer.param_groups[1]['weight_decay'] to set weight decay
    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        assert device.type == 'cuda'
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        try:
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        except (AttributeError, TypeError):
            # fallback to CUDA only AMP for PyTorch < 1.10
            assert device.type == 'cuda'
            amp_autocast = torch.cuda.amp.autocast
        if device.type == 'cuda' and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            _logger.info('AMP not enabled. Training in float32.')
    # optionally resume from a checkpoint

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        _logger.info(f'Using model EMA with decay={args.model_ema_decay}')
        model_ema = utils.ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        ema_decay = cosine_scheduler(args.model_ema_decay, 1, args.epochs)
    
    resume_epoch = None
    if args.early_stopping:
        early_stopping = EarlyStopper(patience=args.patience_epochs, min_delta=0)
        _logger.info(f"Early stopping enabled with patience {args.patience_epochs}")
    else:
        early_stopping = None
    best_metric = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
            dino_loss=args.dino_loss,
            early_stopping=early_stopping,
        )

        if os.path.exists(os.path.join(output_dir, 'summary.csv')):
            best_metric = extract_best_epochs_from_summary(os.path.join(output_dir, 'summary.csv'), args.eval_metric)
        if best_metric is not None:
            early_stopping.validation_accuracy = best_metric

        if args.model_ema:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    if args.torchcompile:
        # torch compile should be done after DDP
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        model = torch.compile(model, backend=args.torchcompile)

    # create the train and eval datasets
    if args.data and not args.data_dir:
        args.data_dir = args.data
    # Set the transform mode to train or test for Random or CenterCrop
    if not args.external_transform:
        # Used for training CoDE
        transform_train = MassiveTransform(args)
        transform_train.mode = 'train'
        transform_test = MassiveTransform(args)
        transform_test.mode = 'test'
    else:
        transform_train = PipelineAugmentation(args, mode='train')
        transform_test = PipelineAugmentation(args, mode='test')

    # transform = LowTransform(args)
    loader_train, dataset_train = create_loader_webdataset(args, shard=args.train_shards, dataset_path=args.data_dir,
                                                           resampling=True,
                                                           double_contrastive=args.double_contrastive,
                                                           generator=args.data_generator, transform=transform_train,
                                                           data_len=args.data_len_train, batch_size=args.batch_size,
                                                           shuffle=True, process_type=args.dataset,
                                                           use_transform=False if args.no_data_augmentation else True, workers=args.workers,
                                                           training_step=args.num_step)
    loader_eval, dataset_eval = create_loader_webdataset(args, shard=args.val_shards_augm, dataset_path=args.data_dir_eval_augm,
                                                         resampling=False,
                                                         double_contrastive=args.double_contrastive,
                                                         generator=args.data_generator, transform=transform_test,
                                                         data_len=args.data_len_eval,
                                                         batch_size=args.validation_batch_size,
                                                         shuffle=False, process_type=args.dataset, use_transform=False,
                                                         workers=args.workers_validate)
    loader_eval_augm, dataset_eval_augm = create_loader_webdataset(args, shard=args.val_shards_augm,
                                                                   dataset_path=args.data_dir_eval_augm,
                                                                   resampling=False,
                                                                   double_contrastive=args.double_contrastive,
                                                                   generator=args.data_generator, transform=transform_test,
                                                                   data_len=args.data_len_eval,
                                                                   batch_size=args.validation_batch_size, shuffle=False,
                                                                   process_type=args.dataset_eval, use_transform=False,
                                                                   workers=args.workers_validate)
    loader_eval_no_augm, dataset_eval_no_augm = create_loader_webdataset(args, shard=args.val_shards_no_augm,
                                                                         dataset_path=args.data_dir_eval_no_augm,
                                                                         resampling=False,
                                                                         double_contrastive=args.double_contrastive,
                                                                         generator=args.data_generator,
                                                                         transform=transform_test,
                                                                         data_len=args.data_len_eval,
                                                                         batch_size=args.validation_batch_size,
                                                                         shuffle=False,
                                                                         process_type=args.dataset_eval,
                                                                         use_transform=False,
                                                                         workers=args.workers_validate)

    loader_train_metric, dataset_train_metric = create_loader_webdataset(args, shard=args.linear_train_shards,
                                                                         dataset_path=args.data_dir,
                                                                         resampling=False,
                                                                         data_len=args.data_len_linear,
                                                                         double_contrastive=args.double_contrastive,
                                                                         generator=args.data_generator,
                                                                         transform=transform_train,
                                                                         batch_size=args.validation_batch_size,
                                                                         shuffle=False,
                                                                         process_type=args.dataset_eval,
                                                                         use_transform=False,
                                                                         workers=args.workers_validate)

    # implement list of all possible losses
    train_loss_fn = InfoNCE(temperature=args.infonce_loss_temperature)

    _logger.info(f"Loss function: {train_loss_fn}")
    train_loss_fn = train_loss_fn.to(device=device)
    # validate_loss_fn = nn.CrossEntropyLoss().to(device=device)
    validate_loss_fn = train_loss_fn.to(device=device)
    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None


    if utils.is_primary(args):
        decreasing = True if eval_metric == 'loss' else False
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    # setup learning rate schedule and starting epoch
    updates_per_epoch = args.num_step
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch

    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        _logger.info(
            f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')
    eval_classifiers = []
    for element in args.classifier:
        classifier = Model(model_name=element, model_path="")
        eval_classifiers.append(classifier)
    classifiers = Models(models=eval_classifiers)

    try:
        if args.only_validate:
            num_epochs = start_epoch + 1
        for epoch in range(start_epoch, num_epochs):
            if args.model_ema:
                model_ema.decay = ema_decay[epoch]
                _logger.info(f"Set EMA decay to {model_ema.decay}")
            if args.weight_decay_end is not None:
                _logger.info(f"Set weight decay to {weight_decay_schedule[epoch]}")
                optimizer.param_groups[1]['weight_decay'] = weight_decay_schedule[epoch]
            if not args.only_validate:
                train_metrics = train_one_epoch(
                    epoch,
                    model,
                    loader_train,
                    optimizer,
                    train_loss_fn,
                    args,
                    lr_scheduler=lr_scheduler,
                    saver=saver,
                    output_dir=output_dir,
                    amp_autocast=amp_autocast,
                    loss_scaler=loss_scaler,
                    model_ema=model_ema,
                )
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    if utils.is_primary(args):
                        _logger.info("Distributing BatchNorm running means and vars")
                    utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')
                eval_metrics = validate(
                    model,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    epoch=epoch,
                    amp_autocast=amp_autocast,
                    model_ema=model_ema,
                )
                print(f"Finished validation {torch.distributed.get_rank()}")
            
            trained_classifiers, dirs = classifier_training(dataloader=loader_train_metric, model=model,
                                                                eval_model=classifiers, output_dir=output_dir, epoch=epoch,
                                                                args=args, distributed=True)
            
            test_features_no_augm, test_labels_no_augm, linear_results = classifier_eval(dataloader=loader_eval_no_augm,
                                                                         classifiers=trained_classifiers, model=model,
                                                                         output_dir=output_dir, epoch=epoch, args=args,
                                                                         distributed=True, transf=False, dirs=dirs)
            test_features_augm, test_labels_augm, linear_results_augm = classifier_eval(dataloader=loader_eval_augm,
                                                                   classifiers=trained_classifiers, model=model,
                                                                   output_dir=output_dir, epoch=epoch, args=args,
                                                                   distributed=True, transf=True, dirs=dirs)
            if epoch % args.plot_freq == 0:
                visualization(loader_eval_no_augm, model, output_dir, epoch, args, test_features=test_features_no_augm,
                              test_labels=test_labels_no_augm)  # w/o transform
                visualization(loader_eval_augm, model, output_dir, epoch, args, transf=True,
                              test_features=test_features_augm, test_labels=test_labels_augm)  # with transform
                dist.barrier()
            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                # remove metrics only on ema model
                # ema_eval_metrics = validate(
                #     model_ema.module,
                #     loader_eval,
                #     validate_loss_fn,
                #     args,
                #     amp_autocast=amp_autocast,
                #     log_suffix=' (EMA)',
                #     epoch=epoch
                # )
                # eval_metrics = ema_eval_metrics
            if args.only_validate:
                train_metrics = OrderedDict([('loss', 0)])
                eval_metrics = OrderedDict([('loss', 0)])

            if utils.is_primary(args):
                accuracy = linear_results[f'linear_tot_classifier_epoch-{epoch}']['maccuracy']
                # broadcast to other process
                if args.distributed:
                    _logger.info(f"Broadcasting accuracy to other process {accuracy}")
                    torch.distributed.broadcast(torch.tensor(accuracy).to(device), 0)
            else:
                accuracy = torch.tensor(0, dtype=torch.float32).to(device)
                if args.distributed:
                    torch.distributed.broadcast(accuracy, 0)
                    print(f"Received accuracy {accuracy}, rank {torch.distributed.get_rank()}")
                accuracy = accuracy.cpu().item()
                eval_metrics['accuracy'] = accuracy

            if utils.is_primary(args):
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                accuracy = linear_results[f'linear_tot_classifier_epoch-{epoch}']['maccuracy']
                eval_metrics['accuracy'] = accuracy
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=not os.path.exists(os.path.join(output_dir, 'summary.csv')),
                    log_wandb=args.log_wandb and has_wandb,
                )
            if args.patience_epochs > 0 and early_stopping.early_stop(accuracy):
                _logger.info(f"Early stopping at epoch {epoch}")
                break

            if saver is not None and not args.only_validate:
                # save proper checkpoint with eval metric
                args.patience_counter = early_stopping.counter
                _logger.info(f"Early stopping counter {args.patience_counter}")
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            if lr_scheduler is not None:
                # step LR for next epoch
                # TODO: check the learning rate schedule because we should have this on engine script
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])



    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


if __name__ == '__main__':
    main()
