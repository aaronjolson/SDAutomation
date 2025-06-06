from i2i_adetailer_fix_hands_only_workflow import i2i_adetailer_fix_hands_only_workflow
from constants import WEBUI_SERVER_URL
from core import change_model, encode_file_to_base64
from image_metadata_reader_exiftool import get_image_parameters
from utils import get_image_files


hand = "perfect female (hand), detailed, perfection "
# h = '<lora:hand4:1.0>'
h = '<lora:hand55:1.0>'

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
hand_prompt = f"{hand},{suffix}"

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

    i2i_adetailer_fix_hands_only_workflow(
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
        distilled_cfg_scale=3.5,
        denoising_strength=0.6,
        sampler_name="Euler",
        scheduler="Simple",
        ad_inpaint_width=1024,
        ad_inpaint_height=1024,
        ad_use_steps=True,
        ad_steps=64,
        )


webui_server_url = WEBUI_SERVER_URL
GENERATIONS_PER_IMAGE = 1

t2i_model_name = 'jibMixFlux_v5ItsAliveQ4GGUF'
change_model(webui_server_url, t2i_model_name)

directory_path = "E:\\Stable_diffusion_projects\\Inspiration\\Ice_Magica\\test"

images_list = get_image_files(directory_path)


for image in images_list:
    for i in range(GENERATIONS_PER_IMAGE):

        raw_path = image.replace('\\\\', '\\')  # standard to raw
        images = [
            encode_file_to_base64(raw_path),
        ]

        params = get_image_parameters(image)
        width, height = tuple(map(int, params["Size"].split('x')))  # pull size data from dict

        prompt = f'{params["prompt"]}'

        prompt = prompt.replace(',,', ',').replace(', ,', ',').replace('score_9', '').replace('score_8_up', '').replace('score_7_up', '').replace('score_6_up', '').replace('score_5_up', '')
        negative_prompt = f'{params["negative_prompt"]}'
        # breakpoint()
        wrap(t2i_model_name,
             prompt=prompt,
             negative_prompt=negative_prompt,
             hand_prompt=hand_prompt,
             source_image=images,
             steps=35,
             width=width*1.5,
             height=height*1.5
             )


print("Job Completed Successfully...")


