from i2i_adetailer_face_hand_workflow import i2i_adetailer_face_hand_workflow
from constants import WEBUI_SERVER_URL
from core import change_model, encode_file_to_base64
from image_metadata_reader_exiftool import get_image_parameters
from utils import get_image_files
import re

hands_lora = "<lora:FluxDetailed_Hands3:0.8>"
hand = "perfect detailed hand, perfection "

# d = ',<lora:extremely_detailed:0.5>,'
r = '<lora:zy_Realism_Enhancer_v1:0.4>'
d = '<lora:add-detail-xl:0.5>'

dxl = 'detailedxl'
dxl_lora = "<lora:detailxl:0.5>"

d3t = 'd3t41l3d'
d3t_lora = '<lora:d3t41l3dXLP:0.5>'

rm_lora = '<lora:RMSDXL_Enhance:0.5>'

fr = '<lora:flux_realism_lora:0.7>'

suffix = "photo realistic:1.4, realistic skin:1.4, realistic, realism, ultra detailed, high quality "

def wrap(
    model_name,
    prompt,
    negative_prompt,
    hand_prompt,
    source_image,
    steps,
    width,
    height,
    ):

    if not hand_prompt:
        hand_prompt = prompt

    i2i_adetailer_face_hand_workflow(
        model_name,
        prompt,
        negative_prompt,
        prompt,
        hand_prompt,
        source_image,
        steps=steps,
        width=width,
        height=height,
        cfg_scale=1,
        distilled_cfg_scale=4,
        denoising_strength=0.45,
        sampler_name="[Forge] Flux Realistic",
        # sampler_name="DEIS",
        scheduler="Beta",
        ad_inpaint_width=1024,
        ad_inpaint_height=1024,
        ad_use_steps=True,
        ad_steps=64,
        )

webui_server_url = WEBUI_SERVER_URL
GENERATIONS_PER_IMAGE = 2

t2i_model_name = 'jibMixFlux_v61RealPixFixed'
change_model(webui_server_url, t2i_model_name)

directory_path = "M:\\Stable_diffusion_projects\\Inspiration\FLUX\\upscale\\env\\Upscale\\upscaled"

images_list = get_image_files(directory_path)

hand_prompt = f"{hand},{hands_lora},{suffix}"


for image in images_list:
    for i in range(GENERATIONS_PER_IMAGE):

        raw_path = image.replace('\\\\', '\\')  # standard to raw
        images = [
            encode_file_to_base64(raw_path),
        ]

        params = get_image_parameters(image)
        print(params)
        if params.get("Size"):
            width, height = tuple(map(int, params["Size"].split('x')))  # pull size data from dict
        else:
            width, height = tuple(map(int, params["Image Size"].split('x')))  # pull size data from dict

        prompt = f'{params["prompt"]}'

        prompt = prompt.replace('score_9', '').replace('score_8_up', '').replace('score_7_up', '').replace('score_6_up', '').replace('score_5_up', '')
        remove_lora_pattern = r'<lora:[A-Za-z0-9_\-:\.]+>'
        prompt = re.sub(remove_lora_pattern, '', prompt)

        prompt = prompt.replace(',,', ',').replace(', ,', ',')

        remove_extra_commas_pattern = r',\s*,+'
        prompt = re.sub(remove_extra_commas_pattern, ',', prompt)

        # negative_prompt = f'{params["negative_prompt"]}'

        wrap(t2i_model_name,
             prompt=prompt,
             negative_prompt='',
             hand_prompt=hand_prompt,
             source_image=images,
             steps=36,
             width=width*1.5,
             height=height*1.5
             )

print("Job Completed Successfully...")
