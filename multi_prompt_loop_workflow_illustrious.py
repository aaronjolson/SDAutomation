import sys
import os
import jsonlines

from t2i_adetailer_face_hand_workflow import t2i_adetailer_face_hand_workflow
from constants import WEBUI_SERVER_URL
from core import change_model
from utils import save_progress, load_progress, clear_progress, get_progress_summary

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

# Progress tracking functions
PROGRESS_FILE = "illustrious_workflow_progress.json"


big_prompt_list = get_prompts()

# negative_prompt = """furry, source_pony, 3D, dutch angle, censored, watermark, jpeg artifacts, muscular, ugly, lowres, bad anatomy, extra limb, missing limbs, deformed hands, deformed fingers"""

suffix_real = "photo realistic:1.4, realistic skin:1.4, ultra detailed, high quality, uncensored, realistic, realism"

#  flat colors, retro vibes, anime screencap
# prefix_anime = "masterpiece, Anime, 2d, absurdres, Seinen, anime screencap, anime coloring, very awa"
prefix_anime = "masterwork, masterpiece, best quality, detailed, depth of field, high detail"
# prefix_anime = "masterwork, masterpiece, best quality, detailed, depth of field, high detail"
# masterpiece,best quality,amazing quality,very aesthetic,high resolution,ultra detailed,perfect details,
suffix_anime = "dynamic pose, dynamic angle, anime screencap, very aesthetic, official art, stylized, vibrant, highly detailed, 8k, adult, aged up, absurdres, Seinen, Anime, 2d"
# suffix_anime = "dynamic pose, dynamic angle,  very aesthetic, official art, stylized, vibrant, highly detailed, 8k, adult, aged up"
# suffix_anime = "dynamic pose, dynamic angle, realistic anime style, Flatline, Flat vector illustration, 2d, very aesthetic, official art, stylized, vibrant, digital, best quality, amazing quality, highly detailed, high resolution, perfect details, ultra-detailed, sharp lineart"
# negative_prompt_anime = "chibi, young, child, low detail, text, wet, 3d, painting, crayon, graphite, sketch, photo, jpeg artifacts, signature, cartoon, watermark, worst quality, abstract, glitch, deformed, mutated, ugly, disfigured, long body, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped"
negative_prompt_anime = "lowres, worst quality, low quality, bad anatomy, bad hands, jpeg artifacts, signature, watermark, text, logo, artist name, extra digits, censored, patreon username, loli, wet, sketch"
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
# motion = "dynamic pose, spanning pose,aura, foreshortening, blood, fighting stance, battle, motion lines, motion blur, sword, holding two swords, steel sword,, reaching towards viewer, from side, <lora:NovaIllustrious:1>"
motion = "dynamic pose, spanning pose,aura, foreshortening, fighting stance, battle, motion lines, motion blur, reaching towards viewer"
jk_cinematic = "JK, JK Style, solo, <lora:JK:1.0>"
# glowing = "glowing, aura, glowing particles, light particles, <lora:DK_glowing_style:0.3>"
# glowing = "glowing, aura, glowing particles, glowing flame, light particles, <lora:DK_glowing_style:0.2>"
glowing = "glowing, aura, glowing particles, light particles, <lora:DK_glowing_style:0.33>"
vec = "vector art"
other = "anime-style, semi-realistic, ultra-detailed, sharp lineart, high-resolution, cinematic lighting, high-contrast colors."
mema_flat_style = "<lora:MeMaXL4 Type D:0.75>"

rimix = "<lora:rimixO:0.8>"

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

    try:
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
        print("  ✓ Image generated successfully")
    except Exception as e:
        print(f"  ✗ Error generating image: {e}")
        raise

webui_server_url = WEBUI_SERVER_URL
IMAGES_PER_MODEL = 6

t2i_model_name = 'littleoctopusmixMF_v10'
# t2i_model_name = 'aMixIllustrious_aMix'

# change_model(webui_server_url, t2i_model_name)

# big_prompt_list = []
# with jsonlines.open('outputs_short.jsonl') as reader:
#     for obj in reader:
#         # print(obj['prompt'])
#         big_prompt_list.append(obj['prompt'])

# Load previous progress if available
previous_progress = load_progress()
start_index = 0

if previous_progress:
    response = input(f"Resume from prompt {previous_progress['current_prompt_index'] + 1}/{previous_progress['total_prompts']}? (y/n): ").strip().lower()
    if response == 'y':
        start_index = previous_progress['current_prompt_index']
        print(f"Resuming from prompt {start_index + 1}")
    else:
        print("Starting from the beginning")

total_prompts = len(big_prompt_list)
print(f"Processing {total_prompts} prompts starting from index {start_index}")

try:
    for prompt_index, prompt in enumerate(big_prompt_list[start_index:], start=start_index):
        print(f"\n=== Processing prompt {prompt_index + 1}/{total_prompts} ===")
        print(f"Prompt: {prompt[:100]}...")  # Show first 100 chars of prompt
        
        # Save progress at the start of each prompt
        save_progress(prompt_index, total_prompts, prompt)

        # prompt_mod = f"{prefix_anime},{prompt},{suffix_anime}"
        # prompt_mod = f"{prefix_anime},{prompt},{suffix_anime},{rimix}"
        prompt_mod = f"{prefix_anime},{prompt},{suffix_anime},{mema_flat_style},{express}"
        # prompt_mod = f"{prefix_anime},{prompt},{suffix_anime},{mema_flat_style},{express},{glowing},{motion}"
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
        #          steps=38
        #          )

        print("Processing with GreenMagica...")
        change_model(webui_server_url, 'illustriousMagica_GreenMagica')
        save_progress(prompt_index, total_prompts, prompt, "illustriousMagica_GreenMagica")
        for i in range(IMAGES_PER_MODEL):
            wrap('',
                 prompt_mod,
                 # negative_prompt,
                 negative_prompt_anime,
                 hand_prompt=hand_prompt,
                 steps=40
                 )

        print(f"Completed prompt {prompt_index + 1}/{total_prompts}")

except KeyboardInterrupt:
    print("\nProcess interrupted by user. Progress has been saved.")
    print(f"Resume from prompt {prompt_index + 1} next time.")
except Exception as e:
    print(f"\nError occurred: {e}")
    print(f"Progress saved. Resume from prompt {prompt_index + 1} next time.")
    raise

# Clear progress file when completely done
clear_progress(PROGRESS_FILE)


print("Job Completed Successfully...")
