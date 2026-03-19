MODEL_SIZE="base"

DATASET_DIR="$HOME/repos/EVA-X/data/"
CKPT_DIR="checkpoints/eva_x_${MODEL_SIZE}_patch16_merged520k_mim.pt"
SAVE_DIR="./output/chexpert/vit_${MODEL_SIZE}_eva_x_chexpert_lateral_chexpert5_thursday"

# Build a patient-level lateral-only split so validation isn't tiny.
TRAIN_LIST="$DATASET_DIR/laterals/train.csv"
VAL_LIST='./'  # not used by train.py
TEST_LIST="$DATASET_DIR/laterals/valid.csv"  # used as validation set during training

NUM_GPUS=1 # was 4
BATCH_SIZE=256 # was 256
ACCUM_ITER=4 # gradient accumulation steps 
EPOCHS=60
NUM_WORKERS=1 # was 8

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
    --checkpoint_type "" \
    --epochs ${EPOCHS} \
    --blr 1e-3 --layer_decay 0.55 --weight_decay 0.05 \
    --fixed_lr \
    --model "eva02_base_patch16_xattn_fusedLN_NaiveSwiGLU_subln_RoPE" \
    --warmup_epochs 0 \
    --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
    --data_path ${DATASET_DIR} \
    --num_workers ${NUM_WORKERS} \
    --train_list ${TRAIN_LIST} \
    --val_list ${VAL_LIST} \
    --test_list ${TEST_LIST} \
    --nb_classes 5 \
    --eval_interval 1 \
    --find_unused_parameters \
    --use_mean_pooling \
    --master_port=29500