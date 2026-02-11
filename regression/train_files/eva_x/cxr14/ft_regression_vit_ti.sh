DATASET_DIR='../classification/datasets/cxr14/images'
CKPT_DIR='../classification/checkpoints/eva_x_tiny_patch16_merged520k_mim.pt'
SAVE_DIR='./output/cxr14/vit_ti_eva_x_regression'
TRAIN_LIST='../regression/datasets/data_splits/cxr14/age_labels_train.txt'
VAL_LIST='../regression/datasets/data_splits/cxr14/age_labels_val.txt'
TEST_LIST='../regression/datasets/data_splits/cxr14/age_labels_test.txt'
NUM_GPUS=2
NUM_CPUS=4

# Run from the regression directory
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --use_env train.py \
    --finetune ${CKPT_DIR} \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 256 \
    --label_mean 46.72015158618534 \
    --label_std 16.60267981756069 \
    --epochs 45 \
    --blr 1e-3 --layer_decay 0.55 --weight_decay 0.05 \
    --model 'eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE' \
    --warmup_epochs 5 \
    --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
    --data_path ${DATASET_DIR} \
    --num_workers ${NUM_CPUS} \
    --train_list ${TRAIN_LIST} \
    --val_list ${VAL_LIST} \
    --test_list ${TEST_LIST} \
    --eval_interval 5 \
    --build_timm_transform \
    --aa 'rand-m6-mstd0.5-inc1' \
    --use_mean_pooling \
    --loss_func mse