mkdir "log"
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --content_dir '../../photo2fourcollection/content' \
    --style_dir '../../photo2fourcollection/style' \
    --print_freq 3000 \
    --save_freq 10 \
    --log_dir 'log' \
    --tensorboardx_logdir 'gated_gan' \
    2>&1 | tee log/log.log