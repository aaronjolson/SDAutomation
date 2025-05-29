import sys
import os

import jsonlines

from t2i_adetailer_face_hand_workflow import t2i_adetailer_face_hand_workflow
from constants import WEBUI_SERVER_URL
from core import change_model

# 1. Get the absolute path of the current script (main.py)
current_script_path = os.path.abspath(__file__)
# 2. Get the directory containing the current script (e.g., .../SDAutomation/)
current_script_dir = os.path.dirname(current_script_path)
# 3. Get the parent directory of the current script's directory
# This is the directory that contains both 'SDAutomation' and 'Prompt-Utils'
parent_dir = os.path.dirname(current_script_dir)
# Alternatively, and perhaps more explicitly for "going up one level":
# parent_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir))
# 4. Construct the path to the 'Prompt-Utils' directory
# It's a sibling, so it's in the 'parent_dir'
path_to_prompt_utils_project = os.path.join(parent_dir, 'Prompt-Utils')
# Add this path to sys.path if it's not already there
if path_to_prompt_utils_project not in sys.path:
    sys.path.append(path_to_prompt_utils_project)
    print(f"Added to sys.path: {path_to_prompt_utils_project}")
else:
    print(f"Already in sys.path: {path_to_prompt_utils_project}")
from get_prompts import get_prompts

big_prompt_list = get_prompts()

# negative_prompt = """furry, source_pony, 3D, dutch angle, censored, watermark, jpeg artifacts, muscular, ugly, lowres, bad anatomy, extra limb, missing limbs, deformed hands, deformed fingers"""

suffix_real = "photo realistic:1.4, realistic skin:1.4, ultra detailed, high quality, uncensored, realistic, realism"

#  flat colors, retro vibes, anime screencap
prefix_anime = "masterpiece, Anime, 2d, absurdres, Seinen, anime screencap, anime coloring"
suffix_anime = "realistic anime style, ultra-detailed, sharp lineart, illustration, very aesthetic, official art, stylized, vibrant, digital, best quality, amazing quality, dynamic pose, dynamic angle, highly detailed"
negative_prompt_anime = "chibi, young, child, low detail, text, wet, 3d, painting, crayon, graphite, sketch, photo, jpeg artifacts, signature, cartoon, watermark, worst quality, abstract, glitch, deformed, mutated, ugly, disfigured, long body, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped"
hand = "perfect female (hand), detailed, perfection "

e = '<lora:Expressive_H-000001:0.6>'
eh = 'expressiveh'
fan = '<lora:Fant5yP0ny:0.5>'

prin = '<lora:princess_xl_v2:0.4>'

safe = 'rating_safe'
questionable = 'rating_questionable'
explicit = 'rating_explicit'

hk = ',<lora:HKStyle_V3-000019:0.8>,'
hks = ',HKStyle,'

retro = "r3tr0, <lora:RetroAnimeILXL:0.8>"
# ildetailer = "<lora:reij-detaILer:0.4>"
velmysticfantasy = "MythAn1m3, <lora:iLLMythAn1m3Style:0.5>"
solo_level_style = "slv50, <lora:solo_leveling_by_readandsign:0.3>"
motion = "dynamic pose, spanning pose,aura, foreshortening, blood, fighting stance, battle, motion lines, motion blur, sword, holding two swords, steel sword,, reaching towards viewer, from side, <lora:NovaIllustrious:1>"
jk_cinematic = "JK, JK Style, solo, <lora:JK:1.0>"
# glowing = "glowing, aura, glowing particles, light particles, <lora:DK_glowing_style:0.3>"
glowing = "glowing, aura, glowing particles, glowing flame, light particles, <lora:DK_glowing_style:0.2>"
vec = "vector art"
other = "anime-style, semi-realistic, ultra-detailed, sharp lineart, high-resolution, cinematic lighting, high-contrast colors."
mema_flat_style = "<lora:MeMaXL4 Type D:0.75>"

phm = '<lora:PHM_style_IL_v3.3:0.8> '

express = "KKSWHNS-Style-V1.0, expressiveKuro-V1.0, <lora:KKSWHNS-Style-V1-illust:0.5>"


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
IMAGES_PER_MODEL = 6

# t2i_model_name = 'littleoctopusmixMF_v10'

# change_model(webui_server_url, t2i_model_name)

# big_prompt_list = []
# with jsonlines.open('outputs_short.jsonl') as reader:
#     for obj in reader:
#         # print(obj['prompt'])
#         big_prompt_list.append(obj['prompt'])

for prompt in big_prompt_list:

    # prompt_mod = f"{prefix_anime},{prompt},{suffix_anime}"
    prompt_mod = f"{prefix_anime},{prompt},{suffix_anime},{mema_flat_style},{express}"
    # prompt_mod = f"{prefix_anime},{prompt},{suffix_anime},{mema_flat_style},{velmysticfantasy}"
    # prompt_mod = f"{prefix_anime},{prompt},{suffix_anime},{velmysticfantasy},{mema_flat_style},{solo_level_style},{glowing}"
    # prompt_mod = f"{prompt},{suffix_anime},{velmysticfantasy}"
    # prompt_mod = f"{prompt},{suffix_anime},{ildetailer}"velmysticfantasy

    # prompt_mod = f"{ prefix}{prompt}{suffix}{e}{sin}{h}"
    # prompt_mod = f"{prefix}{prompt}{suffix}{e}{sin}"
    # prompt_mod = f"{prefix}{prompt}{eh}{suffix}{e}{d}"
    # prompt_mod = f"{prefix}{prompt}{eh}{suffix}{e}{d}"
    # prompt_mod = f"{prefix}{prompt}{suffix}{prin}{r}{d}"
    # prompt_mod = f"{prefix}{prompt}{suffix}{pdz3}{pdzrl}{r}{d}"
    # prompt_mod = f"{prefix}{prompt},{igb},{suffix},{ig},{r},{d}"
    # prompt_mod = f"{prefix},{prompt},{suffix},{eh},{e},{d}"
    # prompt_mod = f"{prefix},{prompt},{suffix},{d}"
    # prompt_mod = f"{prefix}{prompt}{suffix}"

    hand_prompt = f"{hand},{suffix_anime}"

    # for i in range(IMAGES_PER_MODEL):
    #     wrap(t2i_model_name,
    #          prompt_mod,
    #          # negative_prompt,
    #          negative_prompt_anime,
    #          hand_prompt=hand_prompt,
    #          steps=40
    #          )

    change_model(webui_server_url, 'Gembyte_10Amethyst')
    for i in range(IMAGES_PER_MODEL):
        wrap('',
             prompt_mod,
             # negative_prompt,
             negative_prompt_anime,
             hand_prompt=hand_prompt,
             steps=40
             )




print("Job Completed Successfully...")
