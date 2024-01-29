import os

from core import (
    call_txt2img_api_and_save_images,
    call_img2img_api_and_save_images,
    change_model
)

webui_server_url = 'http://127.0.0.1:7861'

out_dir = 'api_out'
out_dir_t2i = os.path.join(out_dir, 'txt2img')
out_dir_i2i = os.path.join(out_dir, 'img2img')
os.makedirs(out_dir_t2i, exist_ok=True)
os.makedirs(out_dir_i2i, exist_ok=True)


def call_txt2img_1(webui_server_url, out_dir_t2i):
    payload = {
        "prompt": "photo of landscape, realistic, photorealism, (best quality:1.1), 35mm",  # extra networks also in prompts
        "negative_prompt" : "EasyNegative, worst quality, low quality, blur, fuzzy lines, graininess, watermarks, logos, patreon",
        "steps": 50,
        "width": 768,
        "height": 768,
        "cfg_scale": 7,
        "sampler_name": "DPM++ 2M Karras",
        "n_iter": 2, # batch count
        "restore_faces": False,

        "enable_hr": False,
        "hr_negative_prompt": "",
        "hr_prompt": "",
        "hr_resize_x": 0,
        "hr_resize_y": 0,
        "hr_scale": 2,
        "hr_second_pass_steps": 0,
        "hr_upscaler": "Latent",
        "batch_size": 1,
        "seed": -1,
        "seed_enable_extras": True,
        "seed_resize_from_h": -1,
        "seed_resize_from_w": -1,
        "subseed": -1,
        "subseed_strength": 0,
        "s_churn": 0,
        "s_min_uncond": 0,
        "s_noise": 1,
        "s_tmax": None,
        "s_tmin": 0,


        # example args for x/y/z plot
        # "script_name": "x/y/z plot",
        # "script_args": [
        #     1,
        #     "10,20",
        #     [],
        #     0,
        #     "",
        #     [],
        #     0,
        #     "",
        #     [],
        #     True,
        #     True,
        #     False,
        #     False,
        #     0,
        #     False
        # ],

        # example args for Refiner and ControlNet
        # "alwayson_scripts": {
        #     "ControlNet": {
        #         "args": [
        #             {
        #                 "batch_images": "",
        #                 "control_mode": "Balanced",
        #                 "enabled": True,
        #                 "guidance_end": 1,
        #                 "guidance_start": 0,
        #                 "image": {
        #                     "image": encode_file_to_base64(r"B:\path\to\control\img.png"),
        #                     "mask": None  # base64, None when not need
        #                 },
        #                 "input_mode": "simple",
        #                 "is_ui": True,
        #                 "loopback": False,
        #                 "low_vram": False,
        #                 "model": "control_v11p_sd15_canny [d14c016b]",
        #                 "module": "canny",
        #                 "output_dir": "",
        #                 "pixel_perfect": False,
        #                 "processor_res": 512,
        #                 "resize_mode": "Crop and Resize",
        #                 "threshold_a": 100,
        #                 "threshold_b": 200,
        #                 "weight": 1
        #             }
        #         ]
        #     },
        #     "Refiner": {
        #         "args": [
        #             True,
        #             "sd_xl_refiner_1.0",
        #             0.5
        #         ]
        #     }
        # },
        # "enable_hr": True,
        # "hr_upscaler": "R-ESRGAN 4x+ Anime6B",
        # "hr_scale": 2,
        # "denoising_strength": 0.5,
        # "styles": ['style 1', 'style 2'],
        # "override_settings": {
        #     'sd_model_checkpoint': "sd_xl_base_1.0",  # this can be used to switch sd model
        # },
    }
    images_saved = call_txt2img_api_and_save_images(webui_server_url, out_dir_t2i, **payload)
    return images_saved


def call_img2img_1(webui_server_url, out_dir_i2i, source_image):
    # init_images = [
    #     encode_file_to_base64(r"B:\path\to\img_1.png"),
    #     # encode_file_to_base64(r"B:\path\to\img_2.png"),
    #     # "https://image.can/also/be/a/http/url.png",
    # ]

    # batch_size = 2
    payload = {
        "prompt": "dog, outside",
        "seed": -1,
        "steps": 30,
        "width": 768,
        "height": 768,
        "denoising_strength": 0.7,
        "n_iter": 2,
        "cfg_scale": 7,
        "init_images": source_image,
        "batch_size": len(source_image)
        # "batch_size": batch_size if len(source_image) == 1 else len(source_image),
        # "mask": encode_file_to_base64(r"B:\path\to\mask.png")
    }
    # if len(init_images) > 1 then batch_size should be == len(init_images)
    # else if len(init_images) == 1 then batch_size can be any value int >= 1
    resp = call_img2img_api_and_save_images(webui_server_url, out_dir_i2i, **payload)
    return resp

if __name__ == '__main__':
    # list_available_models()
    change_model(webui_server_url, 'juggernautXL_v8Rundiffusion')
    images_saved = call_txt2img_1(webui_server_url, out_dir_t2i)
    # for image in images_saved:
    #     call_img2img_1(webui_server_url, out_dir_i2i, [image])
    call_img2img_1(webui_server_url, out_dir_i2i, images_saved)

    # there exist a useful extension that allows converting of webui calls to api payload
    # particularly useful when you wish setup arguments of extensions and scripts
    # https://github.com/huchenlei/sd-webui-api-payload-display