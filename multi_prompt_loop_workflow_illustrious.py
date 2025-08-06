import sys
import os
import jsonlines

from t2i_adetailer_face_hand_workflow import t2i_adetailer_face_hand_workflow
from constants import WEBUI_SERVER_URL
from core import change_model
from utils import save_progress, load_progress, clear_progress

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
from illustrious_prompt_utils import modify_prompt, get_negative_prompt, get_hand_prompt

# Progress tracking functions
PROGRESS_FILE = "illustrious_workflow_progress.json"


big_prompt_list = get_prompts()


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
previous_progress = load_progress(PROGRESS_FILE)
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
        # save_progress(PROGRESS_FILE,prompt_index, total_prompts, prompt)

        # for i in range(IMAGES_PER_MODEL):
        #     wrap(t2i_model_name,
        #          prompt_mod,
        #          # negative_prompt,
        #          negative_prompt_anime,
        #          hand_prompt=hand_prompt,
        #          steps=38
        #          )

        save_progress(PROGRESS_FILE, prompt_index, total_prompts, prompt, t2i_model_name)
        change_model(webui_server_url, 'illustriousMagica_GreenMagica')
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
