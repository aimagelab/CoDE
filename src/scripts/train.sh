#!/bin/bash

# ----- user-editable section ----------
MASTER_ADDR=127.0.0.1   # or the machine’s hostname/IP
MASTER_PORT=29500       # any free TCP port
RDZV_ID=$(date +%s)     # unique job id; timestamp is fine
# --------------------------------------

python -u -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --rdzv_id=$RDZV_ID --rdzv_backend="c10d" --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" --node_rank=0 --max_restarts=4 \
  "train.py" \
  --config "config/config-training.yaml" \
  --dataset "elsa_v2_train_dino" \
  --model "vit_tiny_patch16_224"  --sched "cosine" \
  --double_contrastive \
  --sup_contrastive_loss \
  --dino_crop \
  --experiment "training_code" \
  --epochs 150 \
  --warmup-epochs 5 \
  --lr 0.002 \
  --opt "adamw" \
  --workers 8 \
  --batch-size 256 \
  --output "./runs" \
  --num_transform 2 \
  --contrastive_loss \
  --last-crop \
  --crop-pct 0.9

