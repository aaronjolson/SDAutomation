from t2i_adetailer_face_hand_workflow import t2i_adetailer_face_hand_workflow
from prompt_lists import DARK_FIGURES, MAGE_WOMEN, CHARACTERS, NECROMANCERS, WARRIORS, WARRIOR_GIRL, DEMON_GIRL, BOSS_BATTLE, DARK_WIZARDS, PRINCESS

big_prompt_list = CHARACTERS + MAGE_WOMEN + PRINCESS + DEMON_GIRL + WARRIORS

negative_prompt = """furry, source_pony, score_6_up, score_5, score_4, NSFW, nude, naked, porn, ugly, lowres, bad anatomy, extra limb, missing limbs, deformed hands, deformed fingers, score 1, score 2, score 3"""

prefix = "score_9, score_8_up, score_7_up, zPDXL, "
suffix = "realistic, realism, highly detailed, perfect quality, high quality, photorealistic, perfect hands, perfection"


e = '<lora:Expressive_H-000001:0.8>'
h = '<lora:hand4:0.5>'
d = ',<lora:extremely_detailed:0.8>,'
sin = ",<lora:sinfully_stylish_SDXL>,"
twi = ",<lora:Concept Art Twilight Style SDXL_LoRA_Pony Diffusion V6 XL:0.8>,"
fan = ',<lora:Fant5yP0ny:0.9>,'


fww = '<lora:Fantasy_Wizard__Witches_PonyV2:0.8>,'
hkm = 'hkmagic'
orb = '<lora:XL_Weapon_Orbstaff_-_By_HailoKnight:0.6>,'
os = ',orbstaff,'
boss = ',<lora:XL_boss_battle:0.8>,'
bs = ",BOSSTYLE,"
hk = ',<lora:HKStyle_V3-000019:0.8>,'
hks = ',HKStyle,'


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
        width=1024,
        height=1280,
        cfg_scale=7,
        distilled_cfg_scale=3.5,
        sampler_name="DPM++ SDE",
        scheduler="Karras",
        ad_inpaint_width=1024,
        ad_inpaint_height=1024,
        ad_use_steps=True,
        ad_steps=64,
        )


for prompt in big_prompt_list:
    prompt_mod = f"{prompt}"

    for i in range(10):
        wrap('cyberrealisticPony_v64',
             prompt,
             negative_prompt,
             steps=40
             )


print("Job Completed Successfully...")


