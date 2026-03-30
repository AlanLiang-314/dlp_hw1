# DEVICE=cuda:1 DATALOADER_WORKERS=4 jupyter nbconvert --to notebook --execute /home/sage/dlp_alan/dlp_hw1/unet_scheme_a_weighted_bce_dice_onecycle.ipynb --output run_scheme_a.ipynb
# DEVICE=cuda:1 DATALOADER_WORKERS=4 jupyter nbconvert --to notebook --execute /home/sage/dlp_alan/dlp_hw1/unet_scheme_b_bce_focal_tversky_cosine_restarts.ipynb --output run_scheme_b.ipynb
# DEVICE=cuda:1 DATALOADER_WORKERS=4 jupyter nbconvert --to notebook --execute /home/sage/dlp_alan/dlp_hw1/unet_scheme_c_focal_dice_ema_flip_tta.ipynb --output run_scheme_c.ipynb
DEVICE=cuda:1 DATALOADER_WORKERS=4 jupyter nbconvert --to notebook --execute /home/sage/dlp_alan/dlp_hw1/unet_scheme_d_bce_basic_v5_recovery.ipynb --output run_scheme_d.ipynb
DEVICE=cuda:1 DATALOADER_WORKERS=4 jupyter nbconvert --to notebook --execute /home/sage/dlp_alan/dlp_hw1/unet_scheme_e_bce_basic_lr_scaled_norm.ipynb --output run_scheme_e.ipynb
DEVICE=cuda:1 DATALOADER_WORKERS=4 jupyter nbconvert --to notebook --execute /home/sage/dlp_alan/dlp_hw1/unet_scheme_f_bce_basic_onecycle_ema_tta.ipynb --output run_scheme_f.ipynb
