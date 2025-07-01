import datetime
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress

import torch
from torch.distributed import get_rank, get_world_size
import torchvision.utils
from timm import utils
from timm.models import model_parameters
import wandb
import random
from losses.loss_utils import roll_index, local_sup_contrastive_loss

_logger = logging.getLogger('engine')

# --------------------------
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
DEBUG_BATCHES = 2


def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        args,
        device=torch.device('cuda'),
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=suppress,
        loss_scaler=None,
        model_ema=None,
        mixup_fn=None,
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:  # delete???
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    losses_contrastive_real_m = utils.AverageMeter()
    losses_contrastive_fake_m = utils.AverageMeter()
    losses_dino_m = utils.AverageMeter()
    losses_real_centering_m = utils.AverageMeter()

    model.train()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = args.num_step
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    keys = []
    print(os.path.join(output_dir, f'keys_epoch-{epoch}_process-{torch.distributed.get_rank()}.txt'))
    for batch_idx, element in enumerate(loader):
        if DEBUG and batch_idx == DEBUG_BATCHES:
            print(keys)
            break
        if batch_idx == args.num_step:
            break
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = 1
        # DICTIONARY for reference
        # {'key': None, 'real_image': None, 'fake': None, 'augmented_real': None, 'augmented_fake': None,
        #  'dino_image_real_global': None,
        #  'dino_image_real_local': None, 'dino_image_fake_global': None,
        #  'dino_image_fake_local': None, 'fake_0': None, 'fake_1': None,
        #  'fake_2': None, 'fake_3': None}
        if not args.prefetcher:
            for idx, item in enumerate(element):
                if isinstance(item[0], torch.Tensor):
                    element[idx] = element[idx].to(device)
        key, real, fake, augmented_real, augmented_fake, dino_image_real_global, dino_image_real_local, dino_image_fake_global, dino_image_fake_local, _, _, _, _ = element
        for element in key:
            keys.append(element)
        if args.permutation_real:
            augmented_real = roll_index(augmented_real)
        if args.permutation_fake:
            augmented_fake = roll_index(augmented_fake)
        input = list(
            filter(lambda item: isinstance(item[0], torch.Tensor), [real, fake, augmented_real, augmented_fake]))
        input = torch.cat(input, dim=0)
        input_dino = list(filter(lambda item: isinstance(item[0], torch.Tensor),
                                 [dino_image_real_global, dino_image_real_local, dino_image_fake_global,
                                  dino_image_fake_local]))
        if len(input_dino) > 0:
            input_dino = torch.cat(input_dino, dim=0)
        # multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        def _forward():
            with amp_autocast():
                features = model(input)

                features_real, features_fake, features_augmented_real, features_augmented_fake = torch.split(
                    features, real.size(0), dim=0)  # 4x 64,192
                # sup contrastive loss real and fake + dino elements (CODE)
                dino_features = model(input_dino)
                global_crop_real, local_crop_real, global_crop_fake, local_crop_fake = torch.split(dino_features,
                                                                                                    real.size(0),
                                                                                                    dim=0)  # 4x 64,192

                contrastive_loss_real = (loss_fn(features_real, features_augmented_real,
                                                    features_augmented_fake) + loss_fn(local_crop_real,
                                                                                    global_crop_real,
                                                                                    global_crop_fake)) / 2
                contrastive_loss_fake = (loss_fn(features_fake, features_augmented_fake,
                                                    features_augmented_real) + loss_fn(local_crop_fake,
                                                                                    global_crop_fake,
                                                                                    global_crop_real)) / 2


                real_centering_loss = torch.zeros(1, requires_grad=False).to(device)
                loss_dino = torch.zeros(1, requires_grad=False).to(device)
                # loss = loss_fn(features_real, features_augmented_real, features_fake)
            return contrastive_loss_real, contrastive_loss_fake, loss_dino, real_centering_loss

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        if has_no_sync and not need_update:
            with model.no_sync():
                contrastive_loss_real, contrastive_loss_fake, dino_loss, real_centering_loss = _forward()
                real_centering_loss= (args.lambda_loss*real_centering_loss) # multiply the mse loss for the lambda value

                contrastive_loss_real += real_centering_loss
                contrastive_loss_real += contrastive_loss_fake
                contrastive_loss_real += dino_loss
                _backward(contrastive_loss_real)
        else:
            contrastive_loss_real, contrastive_loss_fake, dino_loss, real_centering_loss = _forward()
            real_centering_loss = (args.lambda_loss * real_centering_loss)  # multiply the mse loss for the lambda value

            loss = contrastive_loss_real + contrastive_loss_real + dino_loss
            loss = loss + real_centering_loss
            _backward(loss)

        if not args.distributed:
            losses_m.update(loss.item() * accum_steps, real.size(0))
        update_sample_count += real.size(0)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model)

        if args.synchronize_step and device.type == 'cuda':
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item() * accum_steps, input.size(0))
                reduced_loss_contrastive_real = utils.reduce_tensor(contrastive_loss_real.data, args.world_size)
                losses_contrastive_real_m.update(reduced_loss_contrastive_real.item() * accum_steps, input.size(0))
                reduced_loss_contrastive_fake = utils.reduce_tensor(contrastive_loss_fake.data, args.world_size)
                losses_contrastive_fake_m.update(reduced_loss_contrastive_fake.item() * accum_steps, input.size(0))
                dino_loss = utils.reduce_tensor(dino_loss.data, args.world_size)
                losses_dino_m.update(dino_loss.item() * accum_steps, real.size(0))
                reduced_real_centering_loss = utils.reduce_tensor(real_centering_loss.data, args.world_size)
                losses_real_centering_m.update(reduced_real_centering_loss.item() * accum_steps, input.size(0))
                update_sample_count *= args.world_size

            if utils.is_primary(args):
                _logger.info(
                    f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                    f'({100. * update_idx / (updates_per_epoch - 1):>3.0f}%)]  '
                    f'Loss contrastive 1: {losses_contrastive_real_m.val:#.3g} ({losses_contrastive_real_m.avg:#.3g})  '
                    f'Loss contrastive 2: {losses_contrastive_fake_m.val:#.3g} ({losses_contrastive_fake_m.avg:#.3g})  '
                    f'Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  '
                    f'Loss dino: {losses_dino_m.val:#.3g} ({losses_dino_m.avg:#.3g})  '
                    f'Loss Real centering: {losses_real_centering_m.val:#.3g} ({losses_real_centering_m.avg:#.3g})  '
                    f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                    f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                    f'LR: {lr:.3e}  '
                    f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True
                    )
        if saver is not None and args.recovery_interval and (
                (update_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()
        # end for
    if args.distributed:
        torch.distributed.barrier()
        print(f"Passed barrier {torch.distributed.get_rank()}")
        print(f"{'sync_lookahead' if hasattr(optimizer, 'sync_lookahead') else 'no_sync_lookahead'}")
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    # with open(os.path.join(output_dir,f'keys_epoch-{epoch}_process-{torch.distributed.get_rank()}.txt'), 'w') as f:
    #    for item in keys:
    #        f.write(str(item) + "\n")
    return OrderedDict([('loss', losses_m.avg), ('loss_contrastive_1', losses_contrastive_real_m.avg),
                        ('loss_contrastive_2', losses_contrastive_fake_m.avg), ('loss_dino', losses_dino_m.avg), ('loss_mse', losses_real_centering_m.avg)])


def validate(
        model,
        loader,
        loss_fn,
        args,
        epoch,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        log_suffix='',
        model_ema=None,
):
    print(f"Entered validate {torch.distributed.get_rank()}")
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    losses_contrastive_real_m = utils.AverageMeter()
    losses_contrastive_fake_m = utils.AverageMeter()
    losses_dino_m = utils.AverageMeter()
    model.eval()
    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        print(f"Entered validate nograd {torch.distributed.get_rank()}")
        for batch_idx, element in enumerate(loader):
            print(f"Fetching validate data rank {torch.distributed.get_rank()}")
            if DEBUG and batch_idx == DEBUG_BATCHES:
                break
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                for idx, item in enumerate(element):
                    if isinstance(item[0], torch.Tensor):
                        element[idx] = element[idx].to(device)
            key, real, fake, augmented_real, augmented_fake, dino_image_real_global, dino_image_real_local, dino_image_fake_global, dino_image_fake_local, _, _, _, _ = element
            if args.permutation_real:
                augmented_real = roll_index(augmented_real)
            if args.permutation_fake:
                augmented_fake = roll_index(augmented_fake)
            input = list(
                filter(lambda item: isinstance(item[0], torch.Tensor), [real, fake, augmented_real, augmented_fake]))
            input = torch.cat(input, dim=0)
            input_dino = list(filter(lambda item: isinstance(item[0], torch.Tensor),
                                     [dino_image_real_global, dino_image_real_local, dino_image_fake_global,
                                      dino_image_fake_local]))
            if len(input_dino) > 0:
                input_dino = torch.cat(input_dino, dim=0)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                features = model(input)
                

                features_real, features_fake, features_augmented_real, features_augmented_fake = torch.split(
                        features, real.size(0), dim=0)  # 4x 64,192
                dino_features = model(input_dino)
                global_crop_real, local_crop_real, global_crop_fake, local_crop_fake = torch.split(dino_features,
                                                                                                    real.size(0),
                                                                                                    dim=0)  # 4x 64,192

                contrastive_loss_real = (loss_fn(features_real, features_augmented_real,
                                                    features_augmented_fake) + loss_fn(local_crop_real,
                                                                                    global_crop_real,
                                                                                    global_crop_fake)) / 2
                contrastive_loss_fake = (loss_fn(features_fake, features_augmented_fake,
                                                    features_augmented_real) + loss_fn(local_crop_fake,
                                                                                    global_crop_fake,
                                                                                    global_crop_real)) / 2
                if not args.dino_loss:
                    loss_dino = torch.zeros(1, requires_grad=False).to(device)

                loss = contrastive_loss_real + contrastive_loss_fake + loss_dino

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                reduced_contrastive_loss_real = utils.reduce_tensor(contrastive_loss_real.data, args.world_size)
                reduced_contrastive_loss_fake = utils.reduce_tensor(contrastive_loss_fake.data, args.world_size)
                reduced_dino_loss = utils.reduce_tensor(loss_dino.data, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == 'cuda':
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), real.size(0))
            losses_contrastive_real_m.update(reduced_contrastive_loss_real.item(), real.size(0))
            losses_contrastive_fake_m.update(reduced_contrastive_loss_fake.item(), real.size(0))
            losses_dino_m.update(reduced_dino_loss.item(), real.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                    f'Loss contrastive 1: {losses_contrastive_real_m.val:#.3g} ({losses_contrastive_real_m.avg:#.3g})  '
                    f'Loss contrastive 2: {losses_contrastive_fake_m.val:#.3g} ({losses_contrastive_fake_m.avg:#.3g})  '
                    f'Loss dino: {losses_dino_m.val:#.3g} ({losses_dino_m.avg:#.3g})  '
                )

    metrics = OrderedDict([('loss', losses_m.avg), ('loss_contrastive_1', losses_contrastive_real_m.avg),
                           ('loss_contrastive_2', losses_contrastive_fake_m.avg), ('loss_dino', losses_dino_m.avg)])
    return metrics
