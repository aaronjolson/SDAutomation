from core import call_img2img_api_and_save_images


def img2img_basic(webui_server_url, out_dir_i2i, source_image,
                  prompt: str, negative_prompt: str = "",
                  steps: int = 30, width: int = 512, height: int = 512, cfg_scale: int = 7,
                  sampler_name: str = "DPM++ 2M Karras", batch_count: int = 1, denoising_strength: float = 0.7,
                  restore_faces: bool = False, enable_hr: bool = False
                  ):

    # alternative methods for proving images for img2img
    # init_images = [
    #     encode_file_to_base64(r"B:\path\to\img_1.png"),
    #     # encode_file_to_base64(r"B:\path\to\img_2.png"),
    #     # "https://image.can/also/be/a/http/url.png",
    # ]

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "width": width,
        "height": height,
        "sampler_name": sampler_name,
        "denoising_strength": denoising_strength,
        "n_iter": batch_count,
        "cfg_scale": cfg_scale,
        "init_images": source_image,
        "batch_size": len(source_image),
        "restore_faces": restore_faces,
        "enable_hr": enable_hr
        # "seed": -1,
        # "batch_size": batch_size if len(source_image) == 1 else len(source_image),
        # "mask": encode_file_to_base64(r"B:\path\to\mask.png")
    }
    # if len(init_images) > 1 then batch_size should be == len(init_images)
    # else if len(init_images) == 1 then batch_size can be any value int >= 1
    resp = call_img2img_api_and_save_images(webui_server_url, out_dir_i2i, **payload)
    return resp
