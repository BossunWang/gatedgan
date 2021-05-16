CUDA_VISIBLE_DEVICES=1 python test.py \
--content_dir '../../WhiteBoxAnimeGAN/test_img' \
--style_dir '../../photo2fourcollection/style/test_train_part' \
--img_scale 3 \
--model_path 'saved_models/GatedGAN_Epoch_9.pt'

#CUDA_VISIBLE_DEVICES=1 python test.py \
#--content_dir '../../custom_pictures' \
#--style_dir '../../photo2fourcollection/style/test_train_part' \
#--img_scale 4 \
#--model_path 'saved_models/GatedGAN_Epoch_9.pt'