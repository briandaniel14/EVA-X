DATASET_DIR='/tmp/rexgrad-val'
CKPT_DIR='/tmp/eva_x_checkpoints/sex_resnet.pth'
SAVE_DIR='./output/rexgradient/r50_mgca_cxr14_sex_eval'
VAL_LIST='datasets/data_splits/rexgradient/sex_labels_valid.txt'
NUM_GPUS=1
NUM_CPUS=4

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --use_env train.py \
    --finetune ${CKPT_DIR} \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 256 \
    --checkpoint_type "" \
    --epochs 45 \
    --blr 1e-3 --weight_decay 0.05 \
    --fixed_lr \
    --model 'resnet50' \
    --warmup_epochs 5 \
    --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
    --data_path ${DATASET_DIR} \
    --num_workers ${NUM_CPUS} \
    --train_list ${VAL_LIST} \
    --val_list ${VAL_LIST} \
    --test_list ${VAL_LIST} \
    --nb_classes 2 \
    --eval_interval 5 \
    --build_timm_transform \
    --aa 'rand-m6-mstd0.5-inc1' \
    --eval
