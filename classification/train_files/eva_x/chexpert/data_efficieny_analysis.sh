#!/bin/bash
# Data efficiency analysis: train lateral CheXpert-5 at 1%, 10%, and 100% of training data.

DATASET_DIR="$HOME/EVA-X/data/CheXpert-v1.0-small/"
CKPT_DIR='checkpoints/eva_x_tiny_patch16_merged520k_mim.pt'
TRAIN_LIST="$DATASET_DIR/laterals/train.csv"
VAL_LIST='./'
TEST_LIST="$DATASET_DIR/laterals/valid.csv"

NUM_GPUS=1
BATCH_SIZE=64
ACCUM_ITER=4
EPOCHS=200
NUM_WORKERS=1

for DATA_PCT in 0.01 0.10 1.0; do
    # Human-readable label for folder name
    if   [[ "$DATA_PCT" == "0.01" ]]; then PCT_LABEL="1pct"
    elif [[ "$DATA_PCT" == "0.10" ]]; then PCT_LABEL="10pct"
    elif [[ "$DATA_PCT" == "1.0"  ]]; then PCT_LABEL="100pct"
    fi

    SAVE_DIR="./output/chexpert/data_efficiency/lateral_${PCT_LABEL}"
    echo "============================================"
    echo "  Training with ${PCT_LABEL} of lateral data"
    echo "  Output: ${SAVE_DIR}"
    echo "============================================"

    OMP_NUM_THREADS=1 python -m torch.distributed.launch \
        --nproc_per_node=${NUM_GPUS} \
        --use_env train.py \
        --dataset chexpert \
        --chexpert_view lateral \
        --chexpert_labels chexpert5 \
        --input_size 224 \
        --finetune ${CKPT_DIR} \
        --output_dir ${SAVE_DIR} \
        --log_dir ${SAVE_DIR} \
        --batch_size ${BATCH_SIZE} \
        --accum_iter ${ACCUM_ITER} \
        --checkpoint_type "" \
        --epochs ${EPOCHS} \
        --blr 5e-4 --layer_decay 0.55 --weight_decay 0.05 \
        --model 'eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE' \
        --warmup_epochs 40 \
        --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
        --data_path ${DATASET_DIR} \
        --data_pct ${DATA_PCT} \
        --num_workers ${NUM_WORKERS} \
        --train_list ${TRAIN_LIST} \
        --val_list ${VAL_LIST} \
        --test_list ${TEST_LIST} \
        --nb_classes 5 \
        --eval_interval 1 \
        --find_unused_parameters \
        --use_mean_pooling \
        --stop_grad_conv1 \
        --build_timm_transform

    echo ""
    echo "  Finished ${PCT_LABEL}. Results saved to ${SAVE_DIR}"
    echo ""
done

echo "All data efficiency runs complete."
