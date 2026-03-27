#!/bin/sh 

source ~/.bashrc 
conda activate evax 
cd ~/repos/EVA-X/classification 
./train_files/eva_x/chexpert/ft_laterals_vit_ti.sh



