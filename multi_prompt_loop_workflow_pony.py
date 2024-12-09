from t2i_adetailer_face_hand_workflow import t2i_adetailer_face_hand_workflow
from constants import WEBUI_SERVER_URL
from prompt_lists import DARK_FIGURES, MAGE_WOMEN, CHARACTERS, NECROMANCERS, WARRIORS, WARRIOR_GIRL, DEMON_GIRL, \
    BOSS_BATTLE, DARK_WIZARDS, PRINCESS, CYBERPUNK, LORA_CHARACTERS, LORA_WARCRAFT, ANGEL_GIRL, FAIRY_GIRL
from core import change_model

big_prompt_list = LORA_CHARACTERS + LORA_WARCRAFT + CHARACTERS

# negative_prompt = """furry, source_pony, 3D, dutch angle, censored, watermark, jpeg artifacts, muscular, ugly, lowres, bad anatomy, extra limb, missing limbs, deformed hands, deformed fingers"""
negative_prompt = """score_6, score_5, score_4, score_1, source_anime, source_pony,  painting, illustration, 3D rendering, CG, ugly, lowres, worst quality, low quality, bad anatomy, extra limb, missing limbs, deformed hands, deformed fingers, signature, watermarks, imperfect eyes, skewed eyes, unnatural face, unnatural body, error, painting by bad-artist, cross-eyed, (lazy eye), bucktooth, censored, watermark, jpeg artifacts, muscular"""

prefix = "(score_9), score_8_up, score_7_up"
suffix = "photo realistic:1.4, realistic skin:1.4, ultra detailed, high quality, uncensored, realistic, realism"
hand = "perfect female (hand), detailed, perfection "
pdz = ',zPDXL,'
pdz3 = ',zPDXL3,'
pdzrl = ',zPDXLrl,'

r = '<lora:zy_Realism_Enhancer_v1:0.4>'
e = '<lora:Expressive_H-000001:0.6>'
eh = 'expressiveh'
h = '<lora:hand4:1.0>'
ed = '<lora:extremely_detailed:0.5>'
d = '<lora:add-detail-xl:0.4>'
sin = '<lora:sinfully_stylish_PONY_02:0.5>'
twi = '<lora:Concept Art Twilight Style SDXL_LoRA_Pony Diffusion V6 XL:0.6>'
fan = '<lora:Fant5yP0ny:0.5>'

prin = '<lora:princess_xl_v2:0.4>'

safe = ',rating_safe,'
questionable = ',rating_questionable,'
explicit = ''

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
    hand_prompt=None,
    steps=40,
    ):

    if not hand_prompt:
        hand_prompt = prompt

    t2i_adetailer_face_hand_workflow(
        model_name,
        prompt,
        negative_prompt,
        prompt,
        hand_prompt,
        steps=steps,
        width=1024,
        height=1280,
        cfg_scale=6,
        distilled_cfg_scale=7,
        sampler_name="DPM++ 2M SDE",
        scheduler="Karras",
        ad_inpaint_width=1024,
        ad_inpaint_height=1024,
        ad_use_steps=True,
        ad_steps=64,
        )

webui_server_url = WEBUI_SERVER_URL
IMAGES_PER_MODEL = 8

t2i_model_name = 'magicaPonyRealism_IceMagia'
change_model(webui_server_url, t2i_model_name)

for prompt in big_prompt_list:
    # prompt_mod = f"{ prefix}{prompt}{suffix}{e}{sin}{h}"
    # prompt_mod = f"{prefix}{prompt}{suffix}{e}{sin}"
    # prompt_mod = f"{prefix}{prompt}{eh}{suffix}{e}{d}"
    # prompt_mod = f"{prefix}{prompt}{eh}{suffix}{e}{d}"
    # prompt_mod = f"{prefix}{prompt}{suffix}{prin}{r}{d}"
    # prompt_mod = f"{prefix}{prompt}{suffix}{pdz3}{pdzrl}{r}{d}"
    # prompt_mod = f"{prefix}{prompt},{igb},{suffix},{ig},{r},{d}"
    # prompt_mod = f"{prefix},{prompt},{suffix},{eh},{e},{d}"
    prompt_mod = f"{prefix},{prompt},{suffix},{d}"
    # prompt_mod = f"{prefix}{prompt}{suffix}"

    hand_prompt = f"{h},{hand},{suffix}"

    for i in range(IMAGES_PER_MODEL):
        wrap(t2i_model_name,
             prompt_mod,
             negative_prompt,
             hand_prompt=hand_prompt,
             steps=40
             )



print("Job Completed Successfully...")


