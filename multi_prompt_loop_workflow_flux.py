import os
import sys
import jsonlines
from core import change_model
from constants import WEBUI_SERVER_URL
from multi_image_loop_i2i_upscale_workflow_fix_hands_only_flux import t2i_model_name
from t2i_adetailer_face_hand_workflow import t2i_adetailer_face_hand_workflow

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
from flux_prompt_utils import modify_prompt, get_negative_prompt, get_hand_prompt

big_prompt_list = get_prompts()


def wrap(
    model_name,
    prompt,
    face_prompt=None,
    hand_prompt=None,
    steps=35,
    ):

    if not hand_prompt:
        hand_prompt = prompt

    t2i_adetailer_face_hand_workflow(
        model_name,
        prompt,
        negative_prompt,
        face_prompt,
        hand_prompt,
        steps=steps,
        width=1408,
        height=1152,
        cfg_scale=1,
        distilled_cfg_scale=4,
        # sampler_name="[Forge] Flux Realistic (2x Slow)",
        sampler_name="[Forge] Flux Realistic",
        # sampler_name="DEIS",
        # sampler_name="Euler",
        # sampler_name="DPM++ 2M",
        scheduler="Beta",
        ad_inpaint_width=1024,
        ad_inpaint_height=1024,
        ad_use_steps=True,
        ad_steps=64,
    )

webui_server_url = WEBUI_SERVER_URL
IMAGES_PER_MODEL = 6
# t2i_model_name = "colossusProjectFlux_v90AIO"
# t2i_model_name = "colossusProjectFlux_v12HephaistosFP8UNET"
# t2i_model_name = "xeHentaiAnimeFlux_04"
# t2i_model_name = "realDream_flux1V1"
# t2i_model_name = "jibMixFlux_v72PixelHeaven"
# t2i_model_name = "jibMixFlux_v85Consisteight"
# t2i_model_name = 'jibMixFlux_v61RealPixFixed'
# t2i_model_name = "splashedFlux_v10"
# t2i_model_name = "fluxmania_Legacy"
# t2i_model_name = "rayflux_v30AIO"
# t2i_model_name = "77FluxAnime_i"

# change_model(webui_server_url, t2i_model_name)

# big_prompt_list = []
# with jsonlines.open('outputs.jsonl') as reader:
#     for obj in reader:
#         big_prompt_list.append(obj['prompt'])

for prompt in big_prompt_list:

    # for i in range(IMAGES_PER_MODEL):
        # prompt_mod = randomify(f"{prompt},{suffix_art}", lora_list)
        # wrap('',
        #      prompt_mod,
        #      face_prompt=face_prompt,
        #      hand_prompt=hand_prompt,
        #      steps=40
        #      )
    for i in range(IMAGES_PER_MODEL):
        change_model(webui_server_url, "fluxKreaFp8_v10")
        prompt_mod = modify_prompt(prompt)
        wrap('',
             prompt_mod,
             face_prompt=prompt,
             hand_prompt=get_hand_prompt(),
             steps=40
             )
    for i in range(IMAGES_PER_MODEL):
        change_model(webui_server_url, "colossusProjectFlux_v90AIO")
        prompt_mod = modify_prompt(prompt)
        wrap('',
             prompt_mod,
             face_prompt=prompt,
             hand_prompt=get_hand_prompt(),
             steps=40
             )
    for i in range(IMAGES_PER_MODEL):
        change_model(webui_server_url, "jibMixFlux_v72PixelHeaven")
        prompt_mod = modify_prompt(prompt)
        wrap('',
             prompt_mod,
             face_prompt=prompt,
             hand_prompt=get_hand_prompt(),
             steps=40
             )
    for i in range(IMAGES_PER_MODEL):
        change_model(webui_server_url, "cyberrealisticFlux_v10FP8")
        prompt_mod = modify_prompt(prompt)
        wrap('',
             prompt_mod,
             face_prompt=prompt,
             hand_prompt=get_hand_prompt(),
             steps=40
             )

    # change_model(webui_server_url, 'magicaPonyRealism_WhiteMagica')

print("Job Completed Successfully...")
