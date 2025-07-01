import argparse
import timm.utils as utils
import yaml

def _parse_args(parser, config_parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def get_args_parser():
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')


    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Dataset parameters
    group = parser.add_argument_group('Dataset parameters')
    # Keep this argument outside the dataset group because it is positional.
    parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                        help='path to dataset (positional is *deprecated*, use --data-dir)')
    parser.add_argument('--data-dir', type=str, default='/work/prin_creative/webdatasets/DeepCOCO',
                        help='path to dataset (root dir)')
    parser.add_argument('--data-dir-eval-augm', type=str, default='/work/prin_creative/webdatasets/DeepCOCO',
                        help='path to dataset (root dir)')
    parser.add_argument('--data-dir-eval-no-augm', type=str, default='/work/prin_creative/webdatasets/DeepCOCO',
                        help='path to dataset (root dir)')
    parser.add_argument('--dataset', metavar='NAME', default='',
                        help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
    parser.add_argument('--dataset-eval',  default='',
                        help='dataset processor for quantitative evaluation (default: same as --dataset)')
    group.add_argument('--train-split', metavar='NAME', default='train',
                       help='dataset train split (default: train)')
    group.add_argument('--only-validate', action="store_true", default=False)
    group.add_argument('--multiple-evaluations', action="store_true", default=False)
    group.add_argument('--val-split', metavar='NAME', default='validation',
                       help='dataset validation split (default: validation)')
    group.add_argument('--train-shards', type=str, default='dataset/shards/coco-training-dict.shards',
                       help='dataset train split')
    group.add_argument('--val-shards-no-augm', type=str, default='dataset/shards/coco-validation-dict.shards',
                       help='dataset validation split')
    group.add_argument('--val-shards-augm', type=str, default='dataset/shards/coco-validation-dict.shards',
                       help='dataset validation split')
    group.add_argument('--test-shards', type=str, default='dataset/shards/coco-test-dict.shards',
                       help='dataset test split')
    group.add_argument('--linear-train-shards', type=str, default='dataset/shards/coco-validation-dict.shards',
                       help='dataset validation split')
    group.add_argument('--val', action='store_true', default = False,
                       help='Just perform evaluation of validation set')
    group.add_argument('--save-model-linear', action='store_true', default=False,
                       help='Save data report on linear evaluation')
    parser.add_argument('--data-len_train', type=int, default= None, help="number of images in the train dataset")
    parser.add_argument('--data-len_eval', type=int, default=4800, help="numnber of images in the eval dataset")
    parser.add_argument('--data-len_linear', type=int, default=9600, help="numnber of images in the linear training dataset")
    parser.add_argument('--num-step', type=int, default=1700, help="numnber of step to change epochs")
    parser.add_argument('--permutation-real', action='store_false', default=True,)
    parser.add_argument('--permutation-fake', action='store_false', default=True,)

    parser.add_argument('--data-generator', type=int, default=None, help="number of generator to be considered")
    group.add_argument('--dataset-download', action='store_true', default=False,
                       help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                       help='path to class to idx mapping file (default: "")')

    # identify the job in the job array
    parser.add_argument('--num_jobs', type=int, default=1)
    parser.add_argument('--job_id', type=int, default=0)

    # transforms parameters
    parser.add_argument('--num_transform', type=int, default=1)

    parser.add_argument("--jitter_min", default=0.5, type=float)
    parser.add_argument("--jitter_max", default=1.5, type=float)
    parser.add_argument("--contrast_min", default=0.5, type=float)
    parser.add_argument("--contrast_max", default=1.5, type=float)
    parser.add_argument("--saturation_min", default=0.5, type=float)
    parser.add_argument("--saturation_max", default=1.5, type=float)
    parser.add_argument("--jpeg_min", default=40, type=int)
    parser.add_argument("--jpeg_max", default=100, type=int)
    parser.add_argument("--opacity_min", default=0.2, type=float)
    parser.add_argument("--opacity_max", default=1.0, type=float)
    parser.add_argument("--resize_min", default=64, type=int)
    parser.add_argument("--resize_max", default=512, type=int)
    parser.add_argument("--scale_min", default=0.5, type=float)
    parser.add_argument("--scale_max", default=1.5, type=float)
    parser.add_argument("--sharp_min", default=1.2, type=float)
    parser.add_argument("--sharp_max", default=2.0, type=float)
    parser.add_argument("--shuffle_min", default=0.0, type=float)
    parser.add_argument("--shuffle_max", default=0.35, type=float)
    parser.add_argument("--skew_min", default=-1.0, type=float)
    parser.add_argument("--skew_max", default=1.0, type=float)
    parser.add_argument("--pad_min", default=0.01, type=float)
    parser.add_argument("--pad_max", default=0.25, type=float)
    parser.add_argument("--brightness-min", default=0.5, type=float)
    parser.add_argument("--brightness-max", default=2, type=float)
    parser.add_argument("--crop-min", default=64, type=float)
    parser.add_argument("--crop-max", default=512, type=float)
    parser.add_argument("--overlay-min", default=0.05, type=float)
    parser.add_argument("--overlay-max", default=0.35, type=float)
    parser.add_argument("--blur-min", default=0.1, type=float)
    parser.add_argument("--blur-max", default=2, type=float)
    parser.add_argument("--ratio-min", default=0.75, type=float)
    parser.add_argument("--ratio-max", default=2, type=float)
    parser.add_argument("--pix-min", default=0.3, type=float)
    parser.add_argument("--pix-max", default=1, type=float)
    parser.add_argument("--rotatio-min", default=90, type=float)
    parser.add_argument("--rotatio-max", default=270, type=float)
    parser.add_argument("--global-crops-scale",type=float, nargs='+', default=(0.4, 1.))
    parser.add_argument("--local-crops-scale",type=float, nargs='+', default=(0.05, 0.4))
    parser.add_argument("--last-crop", action='store_true', default=False)
    parser.add_argument("--random-crop", action='store_true', default=False)

    parser.add_argument('--external-transform', action='store_true', default=False)
    parser.add_argument('--blur_sig', type=float, nargs='+', default=(0, 3.))
    parser.add_argument('--blur_prob', type=float,default=0.01)
    parser.add_argument('--jpeg_prob', type=float,default=0)
    parser.add_argument('--watermark_prob', type=float,default=0.2)
    parser.add_argument('--random_grayscale_prob', type=float,default=0.01)

    # Model parameters
    group = parser.add_argument_group('Model parameters')
    group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                       help='Name of model to train (default: "resnet50")')
    group.add_argument('--pretrained', action='store_true', default=False,
                       help='Start with pretrained version of specified network (if avail)')
    group.add_argument('--linear-pretrained', default=None, type=str,
                       help='Enable triplet loss between data.')
    group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                       help='Initialize model from this checkpoint (default: none)')
    group.add_argument('--resume', default='', type=str, metavar='PATH',
                       help='Resume full model and optimizer state from checkpoint (default: none)')
    group.add_argument('--cineca', action='store_true', default=False, help='path for the cineca cluster')
    group.add_argument('--no-resume-opt', action='store_true', default=False,
                       help='prevent resume of optimizer state when resuming model')
    group.add_argument('--num-classes', type=int, default=None, metavar='N',
                       help='number of label classes (Model default if None)')
    group.add_argument('--gp', default=None, type=str, metavar='POOL',
                       help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    group.add_argument('--img-size', type=int, default=None, metavar='N',
                       help='Image size (default: None => model default)')
    group.add_argument('--in-chans', type=int, default=None, metavar='N',
                       help='Image input channels (default: None => 3)')
    group.add_argument('--input-size', default=None, nargs=3, type=int,
                       metavar='N N N',
                       help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    group.add_argument('--crop-pct', default=None, type=float,
                       metavar='N', help='Input image center crop percent (for validation only)')
    group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                       help='Override mean pixel value of dataset')
    group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                       help='Override std deviation of dataset')
    group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                       help='Image resize interpolation type (overrides model)')
    group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                       help='Input batch size for training (default: 128)')
    group.add_argument('-vb', '--validation-batch-size', type=int, default=1, metavar='N',
                       help='Validation batch size override (default: None)')
    group.add_argument('--channels-last', action='store_true', default=False,
                       help='Use channels_last memory layout')
    group.add_argument('--fuser', default='', type=str,
                       help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
    group.add_argument('--grad-accum-steps', type=int, default=1, metavar='N',
                       help='The number of steps to accumulate gradients (default: 1)')
    group.add_argument('--grad-checkpointing', action='store_true', default=False,
                       help='Enable gradient checkpointing through model blocks/stages')
    group.add_argument('--fast-norm', default=False, action='store_true',
                       help='enable experimental fast-norm')
    group.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
    group.add_argument('--head-init-scale', default=None, type=float,
                       help='Head initialization scale')
    group.add_argument('--head-init-bias', default=None, type=float,
                       help='Head initialization bias value')

    # eccv rebuttal args
    group.add_argument('--no-data-augmentation', action='store_true', default=False,
                       help='training CoDE without data augmentation')
    # only global from scratch --> using an old configuration
    group.add_argument('--local-local', action='store_true', default=False,
                       help='training CoDE with local-local loss')
    group.add_argument('--global-global', action='store_true', default=False,
                       help='training CoDE with global-global loss')
    group.add_argument('--auc-for-rebuttal', action="store_true", default=False,
                       help="This args allows single class fitting for knn and svm max_prediction"
                       )
    group.add_argument('--classifier-checkpoint', type=str, default= "/work/horizon_ria_elsa/runs/svm_oneclass_laion9600/crop_ocsvm_kernel_poly_gamma_auto_nu_0_1.joblib",  help="classifier checkpoint for auc calculation"),

    # scripting / codegen
    scripting_group = group.add_mutually_exclusive_group()
    scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                                 help='torch.jit.script the full model')
    scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                                 help="Enable compilation w/ specified backend (default: inductor).")

    # Optimizer parameters
    group = parser.add_argument_group('Optimizer parameters')
    group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                       help='Optimizer (default: "sgd")')
    group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                       help='Optimizer Epsilon (default: None, use opt default)')
    group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                       help='Optimizer Betas (default: None, use opt default)')
    group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                       help='Optimizer momentum (default: 0.9)')
    group.add_argument('--weight-decay', type=float, default=2e-5,
                       help='weight decay (default: 2e-5)')
    group.add_argument('--weight-decay-end', type=float, default=None,
                       help='end decay for scheduling (default: None)')
    group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                       help='Clip gradient norm (default: None, no clipping)')
    group.add_argument('--clip-mode', type=str, default='norm',
                       help='Gradient clipping mode. One of ("norm", "value", "agc")')
    group.add_argument('--layer-decay', type=float, default=None,
                       help='layer-wise learning rate decay (default: None)')
    group.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

    # Learning rate schedule parameters
    group = parser.add_argument_group('Learning rate schedule parameters')
    group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                       help='LR scheduler (default: "step"')
    group.add_argument('--sched-on-updates', action='store_true', default=False,
                       help='Apply LR scheduler step on update instead of epoch end.')
    group.add_argument('--lr', type=float, default=None, metavar='LR',
                       help='learning rate, overrides lr-base if set (default: None)')
    group.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                       help='base learning rate: lr = lr_base * global_batch_size / base_size')
    group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                       help='base learning rate batch size (divisor, default: 256).')
    group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                       help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
    group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                       help='learning rate noise on/off epoch percentages')
    group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                       help='learning rate noise limit percent (default: 0.67)')
    group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                       help='learning rate noise std-dev (default: 1.0)')
    group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                       help='learning rate cycle len multiplier (default: 1.0)')
    group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                       help='amount to decay each learning rate cycle (default: 0.5)')
    group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                       help='learning rate cycle limit, cycles enabled if > 1')
    group.add_argument('--lr-k-decay', type=float, default=1.0,
                       help='learning rate k-decay for cosine/poly (default: 1.0)')
    group.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                       help='warmup learning rate (default: 1e-6)')
    group.add_argument('--min-lr', type=float, default=0, metavar='LR',
                       help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
    group.add_argument('--epochs', type=int, default=300, metavar='N',
                       help='number of epochs to train (default: 300)')
    group.add_argument('--epochs-classifier', type=int, default=1500,
                       help='number of epochs to train classifier (default: 1500)')
    group.add_argument('--classifier', type=str, default=['linear'] ,nargs='+')
    group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                       help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                       help='manual epoch number (useful on restarts)')
    group.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                       help='list of decay epoch indices for multistep lr. must be increasing')
    group.add_argument('--decay-epochs', type=float, default=90, metavar='N',
                       help='epoch interval to decay LR')
    group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                       help='epochs to warmup LR, if scheduler supports')
    group.add_argument('--warmup-prefix', action='store_true', default=False,
                       help='Exclude warmup period from decay schedule.'),
    group.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                       help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                       help='patience epochs for Plateau LR scheduler (default: 10)')
    group.add_argument('--patience-counter', type=int, default=0, metavar='N',
                       help='counter for initialization of patience epochs for early stopping (default: 0)')
    group.add_argument('--early-stopping', action='store_true', default=False,)
    group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                       help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    group = parser.add_argument_group('Augmentation and regularization parameters')
    group.add_argument('--no-aug', action='store_true', default=False,
                       help='Disable all training augmentation, override other train aug args')
    group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                       help='Random resize scale (default: 0.08 1.0)')
    group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                       help='Random resize aspect ratio (default: 0.75 1.33)')
    group.add_argument('--hflip', type=float, default=0.5,
                       help='Horizontal flip training aug probability')
    group.add_argument('--vflip', type=float, default=0.,
                       help='Vertical flip training aug probability')
    group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                       help='Color jitter factor (default: 0.4)')
    group.add_argument('--aa', type=str, default=None, metavar='NAME',
                       help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    group.add_argument('--aug-repeats', type=float, default=0,
                       help='Number of augmentation repetitions (distributed training only) (default: 0)')
    group.add_argument('--aug-splits', type=int, default=0,
                       help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    group.add_argument('--jsd-loss', action='store_true', default=False,
                       help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    group.add_argument('--triplet_loss', action='store_true', default=False,
                       help='Enable triplet loss between data.')
    group.add_argument('--real-centering-loss', action='store_true', default=False,
                       help='Enable triplet loss between data.')
    group.add_argument('--lambda-loss', type=int , default=10,
                       help='Weight value of the real-centering-loss.')
    group.add_argument('--contrastive_loss', action='store_true', default=False,
                       help='Enable contrastive loss between data.')
    group.add_argument('--dino_loss', action='store_true', default=False,
                       help='Enable dino loss between data.')
    group.add_argument('--dino_loss_weight', type=float, default=0.5,
                       help='Enable dino loss between data.')
    group.add_argument('--dino_head', action='store_true', default=False,
                       help='Enable dino additional projection head.')
    group.add_argument('--head_out', type=int, default=192,
                       help='Dimension of the output of the additional projection head.')
    group.add_argument('--head_hidden_dim', type=int, default=384,
                       help='Hidden dimension of the additional projection head.')
    group.add_argument('--head_bottlenck', type=int, default=256,
                       help='Bottleneck dimension of the additional projection head.')
    group.add_argument('--dino_temp', type=float, default=0.1,
                       help='Enable dino loss between data.')
    group.add_argument('--warmup_teacher_temp', type=float, default=0.04,
                       help='Temperature for global output warmup.')
    group.add_argument('--teacher_temp', type=float, default=0.07,
                       help='Enable dino loss between data.')
    group.add_argument('--teacher_temp_fix', action='store_true', default=False,
                       help='Fix temperature for teacher.')
    group.add_argument('--warmup_teacher_temp_epochs', type=int, default=30,
                       help='Enable dino loss between data.')
    group.add_argument('--n_crops',type=int, default=1,
                       help='Enable dino loss between data.')
    group.add_argument('--infonce_loss_temperature', type=float, default=0.1,
                       help='Temperature for the nce_loss.')
    group.add_argument('--sup_contrastive_loss', action='store_true', default=False,
                       help='Enable contrastive loss between data.')
    group.add_argument('--double_contrastive', action='store_true', default=False,
                       help='Enable contrastive loss between data.')
    group.add_argument('--dino_crop', action='store_true', default=False,
                       help='Enable contrastive loss between data.')
    group.add_argument('--margin', type=int, default=1,
                       help='margin for triplet loss function')
    group.add_argument('--distance', type=str, default=None,
                       help='Type of distance in triplet loss. (default: None)')
    group.add_argument('--bce-loss', action='store_true', default=False,
                       help='Enable BCE loss w/ Mixup/CutMix use.')
    group.add_argument('--bce-target-thresh', type=float, default=None,
                       help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                       help='Random erase prob (default: 0.)')
    group.add_argument('--remode', type=str, default='pixel',
                       help='Random erase mode (default: "pixel")')
    group.add_argument('--recount', type=int, default=1,
                       help='Random erase count (default: 1)')
    group.add_argument('--resplit', action='store_true', default=False,
                       help='Do not random erase first (clean) augmentation split')
    group.add_argument('--mixup', type=float, default=0.0,
                       help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    group.add_argument('--cutmix', type=float, default=0.0,
                       help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                       help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    group.add_argument('--mixup-prob', type=float, default=1.0,
                       help='Probability of performing mixup or cutmix when either/both is enabled')
    group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                       help='Probability of switching to cutmix when both mixup and cutmix enabled')
    group.add_argument('--mixup-mode', type=str, default='batch',
                       help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                       help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    group.add_argument('--smoothing', type=float, default=0.1,
                       help='Label smoothing (default: 0.1)')
    group.add_argument('--train-interpolation', type=str, default='random',
                       help='Training interpolation (random, bilinear, bicubic default: "random")')
    group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                       help='Dropout rate (default: 0.)')
    group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                       help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                       help='Drop path rate (default: None)')
    group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                       help='Drop block rate (default: None)')
    group.add_argument('--step', type=int, default=5, metavar='number_step_transform',
                       help='')

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
    group.add_argument('--bn-momentum', type=float, default=None,
                       help='BatchNorm momentum override (if not None)')
    group.add_argument('--bn-eps', type=float, default=None,
                       help='BatchNorm epsilon override (if not None)')
    group.add_argument('--sync-bn', action='store_true',
                       help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    group.add_argument('--dist-bn', type=str, default='reduce',
                       help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    group.add_argument('--split-bn', action='store_true',
                       help='Enable separate BN layers per augmentation split.')

    # Model Exponential Moving Average
    group = parser.add_argument_group('Model exponential moving average parameters')
    group.add_argument('--model-ema', action='store_true', default=False,
                       help='Enable tracking moving average of model weights')
    group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                       help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    group.add_argument('--model-ema-decay', type=float, default=0.9998,
                       help='decay factor for model weights moving average (default: 0.9998)')

    # Misc
    group = parser.add_argument_group('Miscellaneous parameters')
    group.add_argument('--seed', type=int, default=42, metavar='S',
                       help='random seed (default: 42)')
    group.add_argument('--worker-seeding', type=str, default='all',
                       help='worker seed mode (default: all)')
    group.add_argument('--log-interval', type=int, default=50, metavar='N',
                       help='how many batches to wait before logging training status')
    group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                       help='how many batches to wait before writing recovery checkpoint')
    group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                       help='number of checkpoints to keep (default: 10)')
    group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                       help='how many training processes to use (default: 4)')
    group.add_argument('--workers-validate', type=int, default=2, metavar='N',
                       help='how many training processes to use (default: 2)')
    group.add_argument('--save-images', action='store_true', default=False,
                       help='save images of input bathes every log interval for debugging')
    group.add_argument('--amp', action='store_true', default=False,
                       help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    group.add_argument('--amp-dtype', default='float16', type=str,
                       help='lower precision AMP dtype (default: float16)')
    group.add_argument('--amp-impl', default='native', type=str,
                       help='AMP impl to use, "native" or "apex" (default: native)')
    group.add_argument('--no-ddp-bb', action='store_true', default=False,
                       help='Force broadcast buffers for native DDP to off.')
    group.add_argument('--synchronize-step', action='store_true', default=False,
                       help='torch.cuda.synchronize() end of each step')
    group.add_argument('--pin-mem', action='store_true', default=False,
                       help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    group.add_argument('--no-prefetcher', action='store_true', default=False,
                       help='disable fast prefetcher')
    group.add_argument('--output', default='', type=str, metavar='PATH',
                       help='path to output folder (default: none, current dir)')
    group.add_argument('--experiment', default='', type=str, metavar='NAME',
                       help='name of train experiment, name of sub-folder for output')
    group.add_argument('--eval-metric', default='loss', type=str, metavar='EVAL_METRIC',
                       help='Best metric (default: "loss"')
    group.add_argument('--tta', type=int, default=0, metavar='N',
                       help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    group.add_argument("--local_rank", default=0, type=int)
    group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                       help='use the multi-epochs-loader to save time at the beginning of every epoch')
    group.add_argument('--log-wandb', action='store_true', default=False,
                       help='log training and validation metrics to wandb')
    group.add_argument('--plot-freq', type=int, default=5, metavar='N', help='t-sne every N epochs')
    # wandb setup
    parser.add_argument("--wandb_logging", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", default="contrastive-fake", type=str)
    parser.add_argument("--wandb_entity", default="lorenzo_b_master_thesis", type=str)
    parser.add_argument("--wandb_name", default=None, type=str)
    parser.add_argument("--wandb_id", default=None, type=str)
    parser.add_argument(
        "--wandb_resume", default="allow", choices=["never", "must", "allow"], type=str
    )
    parser.add_argument("--wandb_group", default=None, type=str)
    parser.add_argument("--wandb_notes", default=None, type=str)
    group.add_argument('--deterministic', action='store_true', default=True)
    group.add_argument('--benchmark', action='store_true', default=False)
    return parser, config_parser

