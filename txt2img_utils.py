from core import call_txt2img_api_and_save_images


def txt2img_basic(
        webui_server_url: str, out_dir_t2i: str, prompt: str, negative_prompt: str = "",
        steps: int = 30, width: int = 512, height: int = 512, cfg_scale: int = 7, sampler_name: str = "DPM++ 2M Karras",
        batch_count: int = 1, restore_faces: bool = False, enable_hr: bool = False
):
    payload = {
        "prompt": prompt,  # extra networks also in prompts
        "negative_prompt": negative_prompt,
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler_name,
        "n_iter": batch_count,  # batch count
        "restore_faces": restore_faces,
        "enable_hr": enable_hr
        # "override_settings": {
        #     'sd_model_checkpoint': "sd_xl_base_1.0",  # this can be used to switch sd model
        # }
        # TODO: make separate versions of this function that integrate with other flows such as
        #  txt2img_adetailer, txt2img_controlnet, etc
    }
    images_saved = call_txt2img_api_and_save_images(webui_server_url, out_dir_t2i, **payload)
    return images_saved


def txt2img_adetailer_basic(
        webui_server_url: str, out_dir_t2i: str, prompt: str, negative_prompt: str = "",
        steps: int = 30, width: int = 512, height: int = 512, cfg_scale: int = 7, sampler_name: str = "DPM++ 2M Karras",
        batch_count: int = 1, restore_faces: bool = False, enable_hr: bool = False, ad_model: str = "face_yolov8n.pt",
        ad_inpaint_width: int = 512, ad_inpaint_height: int = 512, ad_use_steps: bool = True, ad_steps: int = 28,
        ad_sampler: str = "DPM++ 2M Karras", ad_prompt: str = "", ad_negative_prompt: str = ""
):
    payload = {
        "prompt": prompt,  # extra networks also in prompts
        "negative_prompt": negative_prompt,
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler_name,
        "n_iter": batch_count,  # batch count
        "restore_faces": restore_faces,
        "enable_hr": enable_hr,
        "alwayson_scripts": {
            "ADetailer": {
                "args": [
                    True,
                    False,
                    {
                        "ad_model": ad_model,
                        "ad_inpaint_width": ad_inpaint_width,
                        "ad_inpaint_height": ad_inpaint_height,
                        "ad_use_steps": ad_use_steps,
                        "ad_steps": ad_steps,
                        "ad_prompt": ad_prompt,
                        "ad_negative_prompt": ad_negative_prompt,
                        "ad_tab_enable": True,
                        "ad_denoising_strength": 0.4,
                        # "ad_model_classes": "",
                        # "ad_confidence": 0.3,
                        # "ad_mask_k_largest": 0,
                        # "ad_mask_min_ratio": 0.0,
                        # "ad_mask_max_ratio": 1.0,
                        # "ad_dilate_erode": 32,
                        # "ad_x_offset": 0,
                        # "ad_y_offset": 0,
                        # "ad_mask_merge_invert": "None",
                        # "ad_mask_blur": 4,
                        # "ad_inpaint_only_masked": True,
                        # "ad_inpaint_only_masked_padding": 0,
                        # "ad_use_inpaint_width_height": False,
                        # "ad_use_cfg_scale": False,
                        # "ad_cfg_scale": 7.0,
                        # "ad_use_checkpoint": False,
                        "ad_checkpoint": "Use same checkpoint",
                        # "ad_use_vae": False,
                        # "ad_vae": "Use same VAE",
                        # "ad_use_sampler": False,
                        # "ad_use_noise_multiplier": False,
                        # "ad_noise_multiplier": 1.0,
                        # "ad_use_clip_skip": False,
                        # "ad_clip_skip": 1,
                        # "ad_restore_face": False,
                        # "ad_controlnet_model": "None",
                        # "ad_controlnet_module": "None",
                        # "ad_controlnet_weight": 1.0,
                        # "ad_controlnet_guidance_start": 0.0,
                        # "ad_controlnet_guidance_end": 1.0
                        "ad_sampler": ad_sampler
                    }
                ]
            }
        }
    }
    # see also
    # https://github.com/Bing-su/adetailer/wiki/REST-API
    images_saved = call_txt2img_api_and_save_images(webui_server_url, out_dir_t2i, **payload)
    return images_saved


def txt2img_adetailer_advanced(
        webui_server_url: str, out_dir_t2i: str, prompt: str, adetailer: list, negative_prompt: str = "",
        steps: int = 30, width: int = 512, height: int = 512, cfg_scale: int = 7,
        sampler_name: str = "DPM++ 2M Karras", batch_count: int = 1, restore_faces: bool = False,
        enable_hr: bool = False,
):
    # if not adetailer_objects:
    #     raise ValueError("You must pass at least 1 valid adetail request object to this function...")

    # args = [True, False,]
    # for obj in adetailer_objects:
    #     args.append(obj)

    payload = {
        "prompt": prompt,  # extra networks also in prompts
        "negative_prompt": negative_prompt,
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler_name,
        "n_iter": batch_count,  # batch count
        "restore_faces": restore_faces,
        "enable_hr": enable_hr,
        "alwayson_scripts": {
            "ADetailer": {
                "args": adetailer
            }
        }
    }
    # print(payload)
    print(payload["alwayson_scripts"]["ADetailer"]["args"])
    # see also
    # https://github.com/Bing-su/adetailer/wiki/REST-API
    images_saved = call_txt2img_api_and_save_images(webui_server_url, out_dir_t2i, **payload)
    return images_saved



        # "batch_size": 1,  # always leave as 1
        # "hr_negative_prompt": "",
        # "hr_prompt": "",
        # "hr_resize_x": 0,
        # "hr_resize_y": 0,
        # "hr_scale": 2,
        # "hr_second_pass_steps": 0,
        # "hr_upscaler": "Latent",

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
