DATASET_DIR='datasets/cxr14/images'
CKPT_DIR='output/cxr14/vit_ti_eva_x_cxr14/checkpoint-best.pth'
SAVE_DIR='./output/cxr14/vit_ti_eva_x_cxr14'
TRAIN_LIST='datasets/data_splits/cxr14/sex_labels_train.txt'
VAL_LIST='datasets/data_splits/cxr14/sex_labels_val.txt'         # not used
TEST_LIST='datasets/data_splits/cxr14/sex_labels_test.txt'
NUM_GPUS=1

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --use_env train.py \
    --finetune ${CKPT_DIR} \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 256 \
    --checkpoint_type "" \
    --epochs 45 \
    --blr 1e-3 --layer_decay 0.55 --weight_decay 0.05 \
    --fixed_lr \
    --model 'eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE' \
    --warmup_epochs 5 \
    --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
    --data_path ${DATASET_DIR} \
    --num_workers 16 \
    --train_list ${TRAIN_LIST} \
    --val_list ${VAL_LIST} \
    --test_list ${VAL_LIST} \
    --nb_classes 2 \
    --eval_interval 5 \
    --build_timm_transform \
    --aa 'rand-m6-mstd0.5-inc1' \
    --use_mean_pooling \
    --eval
