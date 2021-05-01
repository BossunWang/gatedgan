#CUDA_VISIBLE_DEVICES=0 python test.py \
#--content_dir '../../WhiteBoxAnimeGAN/test_img' \
#--style_dir '../../photo2fourcollection/style' \
#--img_scale 3 \
#--model_path 'saved_models_20210328/GatedGAN_Epoch_199.pt'

CUDA_VISIBLE_DEVICES=1 python test.py \
--content_dir '../../custom_pictures' \
--style_dir '../../photo2fourcollection/style/train' \
--img_scale 4 \
--model_path 'saved_models/GatedGAN_Epoch_10.pt'