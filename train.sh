rm -r "log"
mkdir "log"
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --content_dir '../../photo2fourcollection/content' \
    --style_dir '../../photo2fourcollection/style' \
    --print_freq 3000 \
    --epochs 10 \
    --decay_epoch 9 \
    --save_freq 5 \
    --log_dir 'log' \
    --tensorboardx_logdir 'gated_gan'
