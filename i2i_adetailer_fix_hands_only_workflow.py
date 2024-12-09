import os

from img2img_utils import img2img_adetailer_advanced
from constants import WEBUI_SERVER_URL, OUT_DIR

from core import encode_file_to_base64
from image_metadata_reader_exiftool import get_image_parameters


def i2i_adetailer_fix_hands_only_workflow(
        t2i_model_name,
        prompt,
        negative_prompt,
        face_prompt,
        hand_prompt,
        source_image,
        steps=30,
        width=1024,
        height=1280,
        cfg_scale=1,
        distilled_cfg_scale=3.5,
        denoising_strength=0.5,
        sampler_name="Euler",
        scheduler="Simple",
        ad_inpaint_width=1024,
        ad_inpaint_height=1024,
        ad_use_steps=True,
        ad_steps=64,
):
    webui_server_url = WEBUI_SERVER_URL
    out_dir = OUT_DIR
    out_dir_t2i = os.path.join(out_dir, 'img2img')
    os.makedirs(out_dir_t2i, exist_ok=True)

    adetailer = [
        True,
        True,
        {
            "ad_tab_enable": True,
            "ad_model": "hand_yolov8n.pt",
            "ad_inpaint_width": ad_inpaint_width,
            "ad_inpaint_height": ad_inpaint_height,
            "ad_use_steps": ad_use_steps,
            "ad_steps": ad_steps,
            "ad_prompt": hand_prompt,
            "ad_negative_prompt": negative_prompt,
            "ad_denoising_strength": 0.6,
            "ad_model_classes": "",
            "ad_confidence": 0.3,
            "ad_mask_k_largest": 0,
            "ad_mask_min_ratio": 0.0,
            "ad_mask_max_ratio": 1.0,
            "ad_dilate_erode": 32,  # update to 4?
            "ad_x_offset": 0,
            "ad_y_offset": 0,
            "ad_mask_merge_invert": "None",
            "ad_mask_blur": 6,
            "ad_sampler": "Euler",
            "ad_use_sampler": False,
            "ad_inpaint_only_masked": True,
            "ad_inpaint_only_masked_padding": 24,
            "ad_use_cfg_scale": False,
            "ad_cfg_scale": 7.0,
            "ad_use_checkpoint": False,
            "ad_checkpoint": "Use same checkpoint",
            "ad_use_vae": False,
            "ad_vae": "Use same VAE",
            "ad_use_noise_multiplier": False,
            "ad_noise_multiplier": 1.0,
            "ad_use_clip_skip": False,
            "ad_clip_skip": 1,
            "ad_restore_face": False,
            "ad_controlnet_model": "None",
            "ad_controlnet_module": "None",
            "ad_controlnet_weight": 1.0,
            "ad_controlnet_guidance_start": 0.0,
            "ad_controlnet_guidance_end": 1.0
        },
        {"ad_tab_enable": False},
        {"ad_tab_enable": False},
        {"ad_tab_enable": False},
    ]

    img2img_adetailer_advanced(
        webui_server_url,
        out_dir_t2i,
        source_image,
        prompt,
        adetailer,
        negative_prompt=negative_prompt,
        steps=steps,
        width=width,
        height=height,
        cfg_scale=cfg_scale,
        distilled_cfg_scale=distilled_cfg_scale,
        denoising_strength=denoising_strength,
        sampler_name=sampler_name,
        scheduler=scheduler,
        batch_count=1,
        restore_faces=False,
        enable_hr=False
    )


if __name__ == "__main__":
    model_name = "icbinpXL_v6"
    out_dir = OUT_DIR
    out_dir_i2i = os.path.join(out_dir, 'img2img')
    os.makedirs(out_dir_i2i, exist_ok=True)

    path = "E:\\Stable_diffusion_projects\\Inspiration\\Ice_Magica\\txt2img-20241115-120901-0.png"

    raw_path = path.replace('\\\\', '\\')  # standard to raw
    images = [
        encode_file_to_base64(raw_path),
    ]

    params = get_image_parameters(path)
    width, height = tuple(map(int, params["Size"].split('x')))  # pull size data from dict

    for i in range(2):
        i2i_adetailer_fix_hands_only_workflow(model_name,  params["prompt"], params["negative_prompt"], params["prompt"], params["prompt"], images)
