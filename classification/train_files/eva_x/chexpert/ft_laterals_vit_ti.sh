DATASET_DIR="$HOME/EVA-X/data/CheXpert-v1.0-small"
CKPT_DIR='checkpoints/eva_x_tiny_patch16_merged520k_mim.pt'
SAVE_DIR='./output/chexpert/vit_ti_eva_x_chexpert_lateral'

# Build a patient-level lateral-only split so validation isn't tiny.
SPLIT_DIR="$DATASET_DIR/splits/lateral_seed0_val10"
TRAIN_LIST="$SPLIT_DIR/train.csv"
VAL_LIST='./'  # not used by train.py
TEST_LIST="$SPLIT_DIR/val.csv"  # used as validation set during training

if [ ! -f "$TRAIN_LIST" ] || [ ! -f "$TEST_LIST" ]; then
    echo "Generating lateral split into: $SPLIT_DIR"
    python tools/split_chexpert_lateral.py \
        --input_csv "$DATASET_DIR/train.csv" \
        --output_dir "$SPLIT_DIR" \
        --view lateral \
        --val_frac 0.1 \
        --seed 0
fi

NUM_GPUS=1 # was 4
BATCH_SIZE=64 # was 256
ACCUM_ITER=16 # gradient accumulation steps (effective batch = BATCH_SIZE * ACCUM_ITER * NUM_GPUS)
EPOCHS=1 # was 100
NUM_WORKERS=1 # was 8

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --use_env train.py \
    --dataset chexpert \
    --chexpert_view lateral \
    --chexpert_labels chexpert14 \
    --input_size 224 \
    --finetune ${CKPT_DIR} \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size ${BATCH_SIZE} \
    --accum_iter ${ACCUM_ITER} \
    --checkpoint_type "" \
    --epochs ${EPOCHS} \
    --blr 1e-3 --layer_decay 0.55 --weight_decay 0.05 \
    --fixed_lr \
    --model 'eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE' \
    --warmup_epochs 0 \
    --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
    --data_path ${DATASET_DIR} \
    --num_workers ${NUM_WORKERS} \
    --train_list ${TRAIN_LIST} \
    --val_list ${VAL_LIST} \
    --test_list ${TEST_LIST} \
    --nb_classes 14 \
    --eval_interval 1 \
    --find_unused_parameters \
    --use_mean_pooling \
    --stop_grad_conv1 \
    --use_smooth_label
