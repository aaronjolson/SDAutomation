from t2i_adetailer_face_hand_workflow import t2i_adetailer_face_hand_workflow
from prompt_lists import DARK_FIGURES, MAGE_WOMEN, CHARACTERS, NECROMANCERS, WARRIORS, WARRIOR_GIRL, DEMON_GIRL, BOSS_BATTLE, DARK_WIZARDS

big_prompt_list = MAGE_WOMEN + DARK_FIGURES + WARRIORS

prefix = ""
suffix = ""

negative_prompt = ""

def wrap(
    model_name,
    prompt,
    negative_prompt,
    steps=35,
    ):
    t2i_adetailer_face_hand_workflow(
        model_name,
        prompt,
        negative_prompt,
        prompt,
        prompt,
        steps=steps,
        width=1024,
        height=1408,
        cfg_scale=1,
        distilled_cfg_scale=3.5,
        sampler_name="Euler",
        scheduler="Simple",
        ad_inpaint_width=1024,
        ad_inpaint_height=1024,
        ad_use_steps=True,
        ad_steps=64,
        )


for prompt in big_prompt_list:
    prompt_mod = f"{prompt}"

    for i in range(5):
        wrap('jibMixFlux_v5ItsAliveQ4GGUF',
             prompt,
             negative_prompt,
             steps=8
             )

print("Job Completed Successfully...")


