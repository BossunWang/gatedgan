#CUDA_VISIBLE_DEVICES=0 python test.py \
#--content_dir '../../WhiteBoxAnimeGAN/test_img' \
#--style_dir '../../photo2fourcollection/style' \
#--img_scale 3 \
#--model_path 'saved_models_20210328/GatedGAN_Epoch_199.pt'

CUDA_VISIBLE_DEVICES=0 python test.py \
--content_dir '../../custom_pictures' \
--style_dir '../../photo2fourcollection/style' \
--img_scale 6 \
--model_path 'saved_models_20210328/GatedGAN_Epoch_199.pt'