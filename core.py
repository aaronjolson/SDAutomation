from datetime import datetime
import urllib.request
import base64
import json
import time
import os

import requests


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))
        print(f"Image {save_path} written successfully.")


def call_api(webui_server_url, api_endpoint, **payload):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{webui_server_url}/{api_endpoint}',
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    response = urllib.request.urlopen(request)
    print(f"Response from server for job: {data} returned with status {response.getcode()}")
    return json.loads(response.read().decode('utf-8'))


def call_txt2img_api_and_save_images(webui_server_url, out_dir_t2i, **payload):
    response = call_api(webui_server_url, 'sdapi/v1/txt2img', **payload)
    images_saved = []
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_t2i, f'txt2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)
        images_saved.append(image)
    return images_saved


def call_img2img_api_and_save_images(webui_server_url, out_dir_i2i, **payload):
    response = call_api(webui_server_url,'sdapi/v1/img2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_i2i, f'img2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)
    return response


def list_available_models(webui_server_url):
    url = webui_server_url
    response = requests.get(url=f'{url}/sdapi/v1/sd-models')
    print(response.json())


def change_model(webui_server_url, model_name):
    url = webui_server_url
    opt = requests.get(url=f'{url}/sdapi/v1/options')
    opt_json = opt.json()
    opt_json['sd_model_checkpoint'] = model_name
    resp = requests.post(url=f'{url}/sdapi/v1/options', json=opt_json)
    print(f"Response from server changing model to {model_name} : {resp}")
