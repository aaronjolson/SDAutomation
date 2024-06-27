import os

from core import change_model
from txt2img_utils import txt2img_adetailer_advanced
from constants import WEBUI_SERVER_URL, OUT_DIR


def t2i_adetailer_2step_workflow():
    webui_server_url = WEBUI_SERVER_URL
    out_dir = OUT_DIR
    out_dir_t2i = os.path.join(out_dir, 'txt2img')
    os.makedirs(out_dir_t2i, exist_ok=True)

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

    t2i_model_name = "icbinpXL_v6"

    change_model(webui_server_url, t2i_model_name)

    ad_inpaint_width = 768
    ad_inpaint_height = 768
    ad_use_steps = True
    ad_steps = 32
    ad_sampler = "DPM++ 2M Karras"

    adetailer = [
        True,
        False,
        {
            "ad_model": "face_yolov8n.pt",
            "ad_inpaint_width": ad_inpaint_width,
            "ad_inpaint_height": ad_inpaint_height,
            "ad_use_steps": ad_use_steps,
            "ad_steps": ad_steps,
            "ad_prompt": prompt,
            "ad_negative_prompt": negative_prompt,
            "ad_sampler": ad_sampler,
            "ad_tab_enable": True,
            "ad_denoising_strength": 0.4,
        },
        {
            "ad_model": "hand_yolov8n.pt",
            "ad_inpaint_width": ad_inpaint_width,
            "ad_inpaint_height": ad_inpaint_height,
            "ad_use_steps": ad_use_steps,
            "ad_steps": ad_steps,
            "ad_prompt": prompt,
            "ad_negative_prompt": negative_prompt,
            "ad_sampler": ad_sampler,
            "ad_tab_enable": True,
            "ad_denoising_strength": 0.4,
        }
    ]

    txt2img_adetailer_advanced(
        webui_server_url,
        out_dir_t2i,
        prompt,
        adetailer,
        negative_prompt=negative_prompt,
        steps=50,
        width=1024,
        height=1024,
        cfg_scale=7,
        sampler_name="DPM++ 2M Karras",
        batch_count=1,
        restore_faces=False,
        enable_hr=False
    )


if __name__ == "__main__":
    # simple_t2i_adetailer_workflow()
    for i in range(3):
        t2i_adetailer_2step_workflow()