import os

from core import encode_file_to_base64
from constants import WEBUI_SERVER_URL, OUT_DIR
from img2img_utils import img2img_basic
from image_metadata_reader_exiftool import get_image_parameters


def chain(path:str):
    out_dir = OUT_DIR
    out_dir_i2i = os.path.join(out_dir, 'img2img')
    os.makedirs(out_dir_i2i, exist_ok=True)

    raw_path = path.replace('\\\\', '\\') # standard to raw
    images = [
        encode_file_to_base64(raw_path),
    ]

    params = get_image_parameters(path)
    width, height = tuple(map(int, params["Size"].split('x')))  # pull size data from dict

    img2img_basic(WEBUI_SERVER_URL, out_dir_i2i, images, prompt=params["prompt"], negative_prompt=params["negative_prompt"],
                  steps=params["Steps"], width=int(width*1.5), height=int(height*1.5), denoising_strength=0.6)


if __name__ == "__main__":
    path = "E:\\Stable_diffusion_projects\\Inspiration\\Ice_Magica\\txt2img-20241115-120901-0.png"
    chain(path)
