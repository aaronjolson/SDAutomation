from core import change_model
from constants import WEBUI_SERVER_URL
from t2i_adetailer_face_hand_workflow import t2i_adetailer_face_hand_workflow
from prompt_lists import DARK_FIGURES, MAGE_WOMEN, NECROMANCERS, WARRIORS, WARRIOR_GIRL, DEMON_GIRL, \
    DARK_WIZARDS, PRINCESS, CYBERPUNK, ANGEL_GIRL, FAIRY_GIRL
big_prompt_list = MAGE_WOMEN + DARK_FIGURES + WARRIORS + ANGEL_GIRL + FAIRY_GIRL + PRINCESS + CYBERPUNK + DEMON_GIRL + NECROMANCERS

prefix = "Photo of "
suffix = "photo realistic, realistic skin, ultra detailed, cinematic, realism, UHD, highest-quality, intricate details"

negative_prompt = ""

def wrap(
    model_name,
    prompt,
    negative_prompt,
    steps=35,
    ):
    t2i_adetailer_face_hand_workflow(
        model_name,
        prompt,
        negative_prompt,
        prompt,
        prompt,
        steps=steps,
        width=1152,
        height=1408,
        cfg_scale=1,
        distilled_cfg_scale=4,
        # sampler_name="[Forge] Flux Realistic (2x Slow)",
        sampler_name="DPM++ 2M",
        scheduler="Beta",
        ad_inpaint_width=1024,
        ad_inpaint_height=1024,
        ad_use_steps=True,
        ad_steps=64,
        )

webui_server_url = WEBUI_SERVER_URL
IMAGES_PER_MODEL = 5

t2i_model_name = 'jibMixFlux_v61RealPixFixed'
change_model(webui_server_url, t2i_model_name)

for prompt in big_prompt_list:
    # prompt_mod = f"{prefix}{prompt},{suffix}"
    prompt_mod = f"{prompt},{suffix}"

    for i in range(5):
        wrap('',
             prompt_mod,
             negative_prompt,
             steps=36
             )

print("Job Completed Successfully...")


