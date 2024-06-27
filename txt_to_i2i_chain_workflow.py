import os

from core import change_model
from txt2img_utils import txt2img_basic
from img2img_utils import img2img_basic
from constants import WEBUI_SERVER_URL, OUT_DIR


def text_to_image_chain_image_2_image(t2i_model_name: str,  t2i_prompt: str, t2i_negative_prompt: str,
                                      i2i_model_name: str, i2i_prompt: str, i2i_negative_prompt: str, steps: int = 50,
                                      width: int = 768, height: int = 768):
    webui_server_url = WEBUI_SERVER_URL
    out_dir = OUT_DIR
    out_dir_t2i = os.path.join(out_dir, 'txt2img')
    out_dir_i2i = os.path.join(out_dir, 'img2img')
    os.makedirs(out_dir_t2i, exist_ok=True)
    os.makedirs(out_dir_i2i, exist_ok=True)

    change_model(webui_server_url, t2i_model_name)
    images_saved = txt2img_basic(webui_server_url, out_dir_t2i, prompt=t2i_prompt, negative_prompt=t2i_negative_prompt,
                                 steps=steps, width=width, height=height)
    # for image in images_saved:
    #     call_img2img_1(webui_server_url, out_dir_i2i, [image], prompt=i2i_prompt, negative_prompt=i2i_negative_prompt,
    #                   steps=steps, width=width, height=height, denoising_strength=0.7)

    if t2i_model_name != i2i_model_name:
        change_model(webui_server_url, i2i_model_name)
    img2img_basic(webui_server_url, out_dir_i2i, images_saved, prompt=i2i_prompt, negative_prompt=i2i_negative_prompt,
                  steps=steps, width=width, height=height, denoising_strength=0.6)


if __name__ == '__main__':
    text_to_image_chain_image_2_image(
        t2i_model_name='icbinpXL_v6',
        t2i_prompt="photo of landscape, realistic, photorealism, (best quality:1.1)",
        t2i_negative_prompt="low quality, blur, watermarks, logos, patreon",
        i2i_model_name="icbinpXL_v6",
        i2i_prompt="dog outside",
        i2i_negative_prompt="watermarks, logos, patreon"
    )
