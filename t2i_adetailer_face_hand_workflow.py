import os

from txt2img_utils import txt2img_adetailer_advanced
from constants import WEBUI_SERVER_URL, OUT_DIR


def t2i_adetailer_face_hand_workflow(
        t2i_model_name,
        prompt,
        negative_prompt,
        face_prompt,
        hand_prompt,
        steps=30,
        width=1024,
        height=1280,
        cfg_scale=1,
        distilled_cfg_scale=3.5,
        sampler_name="Euler",
        scheduler="Simple",
        ad_inpaint_width=1024,
        ad_inpaint_height=1024,
        ad_use_steps=True,
        ad_steps=64,
):
    webui_server_url = WEBUI_SERVER_URL
    out_dir = OUT_DIR
    out_dir_t2i = os.path.join(out_dir, 'txt2img')
    os.makedirs(out_dir_t2i, exist_ok=True)

    adetailer = [
        True,
        False,
        {
            "ad_tab_enable": True,
            "ad_model": "face_yolov8n.pt",
            "ad_use_inpaint_width_height": True,
            "ad_inpaint_width": ad_inpaint_width,
            "ad_inpaint_height": ad_inpaint_height,
            "ad_use_steps": ad_use_steps,
            "ad_steps": ad_steps,
            "ad_prompt": face_prompt,
            "ad_negative_prompt": negative_prompt,
            "ad_denoising_strength": 0.4,
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
            "ad_inpaint_only_masked_padding": 16,
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
        {
            "ad_tab_enable": True,
            "ad_model": "hand_yolov8n.pt",
            "ad_inpaint_width": 764,
            "ad_inpaint_height": 764,
            "ad_use_steps": ad_use_steps,
            "ad_steps": ad_steps,
            "ad_prompt": hand_prompt,
            "ad_negative_prompt": negative_prompt,
            "ad_denoising_strength": 0.4,
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
            "ad_inpaint_only_masked_padding": 16,
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
        # {"ad_tab_enable": False},
        {"ad_tab_enable": False},
        {"ad_tab_enable": False},

    ]

    txt2img_adetailer_advanced(
        webui_server_url,
        out_dir_t2i,
        prompt,
        adetailer,
        negative_prompt=negative_prompt,
        steps=steps,
        width=width,
        height=height,
        cfg_scale=cfg_scale,
        distilled_cfg_scale=distilled_cfg_scale,
        sampler_name=sampler_name,
        scheduler=scheduler,
        batch_count=1,
        restore_faces=False,
        enable_hr=False
    )


if __name__ == "__main__":
    model_name = "icbinpXL_v6"
    prompt = """
        Character design, full body, 40 years old man, glass, slender, ghost, tall body, dark knight,
        nihilistic, white skin, long white hair, blue iris, medieval fantasy, adventurer clothes, has a big sword,
        good looking, white background, best Quality
        """

    # prompt="""Masterpiece (wide angle shot) , old sorcerer crafting an incantation, (creating a little magic city in a box:1.9),
    # standing on an old carved table in a mage laboratory. (night ambiance:1.6), dark brooding magic look, fantastic view."""

    negative_prompt = """lowres, bad anatomy, bad hands, multiple eyebrow, (cropped), extra limb, missing limbs, 
        deformed hands, long neck, long body, (bad hands), signature, username, artist name, conjoined fingers, 
        deformed fingers, ugly eyes, imperfect eyes, skewed eyes, unnatural face, unnatural body, error
        """

    for i in range(3):
        t2i_adetailer_face_hand_workflow(model_name, prompt, negative_prompt, prompt, prompt)
