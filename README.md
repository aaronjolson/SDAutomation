# Automation
This project is designed to help automate workflows using

## Quickstart
### Run the image gen server in API mode
From inside your SD project dir run
```bash
python launch.py --api --nowebui
```

or alternatively (to also bring up the UI)
```bash
python launch.py --api
```

### Run the desired program workflow
Example
```bash
python txt_to_i2i_chain_workflow.py
```

## Prereqs / Dependencies
In order to run this project you will need
- Python (All testing has been done on version 3.10.*)
- The python requests library (`pip install requests`)
- stable-diffusion-webui - https://github.com/AUTOMATIC1111/stable-diffusion-webui 
- OR stable-diffusion-webui-forge - https://github.com/lllyasviel/stable-diffusion-webui-forge

### Additional
There are development plans to make workflows that include certain popular extensions. 
Check that you have the necessary extensions to run a given workflow before running.

## Additional info
### Read the docs for details on endpoint params
http://127.0.0.1:7860/docs

### Helpful resources 
There exist a useful extension that allows converting of webui calls to api payload
particularly useful when you wish setup arguments of extensions and scripts
https://github.com/huchenlei/sd-webui-api-payload-display
* not sure if this is maintained any longer, or working on the latest versions of the API