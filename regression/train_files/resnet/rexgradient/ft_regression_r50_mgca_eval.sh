DATASET_DIR='/tmp/rexgrad-val'
CKPT_DIR='/tmp/eva_x_checkpoints/r50-log.pth'
SAVE_DIR='./output/rexgradient/resnet50_mgca_regression_eval'
VAL_LIST='./datasets/data_splits/rexgradient/age_labels_valid.txt'
NUM_GPUS=1
NUM_CPUS=4

# Run from the regression directory
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --use_env train.py \
    --finetune ${CKPT_DIR} \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 256 \
    --label_mean 47.66939923712651 \
    --label_std 25.92875001956226 \
    --epochs 75 \
    --blr 2.5e-4 --weight_decay 0.05 \
    --model 'resnet50' \
    --warmup_epochs 5 \
    --drop_path 0 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
    --data_path ${DATASET_DIR} \
    --num_workers ${NUM_CPUS} \
    --train_list ${VAL_LIST} \
    --val_list ${VAL_LIST} \
    --test_list ${VAL_LIST} \
    --eval_interval 10 \
    --min_lr 1e-5 \
    --build_timm_transform \
    --aa 'rand-m6-mstd0.5-inc1' \
    --loss_func mse \
    --mlp_layers 1 \
    --eval
