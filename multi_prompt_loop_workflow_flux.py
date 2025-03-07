import jsonlines

from core import change_model
from constants import WEBUI_SERVER_URL
from multi_image_loop_i2i_upscale_workflow_fix_hands_only_flux import t2i_model_name
from t2i_adetailer_face_hand_workflow import t2i_adetailer_face_hand_workflow
from prompt_lists import DARK_FIGURES, MAGE_WOMEN, NECROMANCERS, WARRIORS, WARRIOR_GIRL, DEMON_GIRL, \
    DARK_WIZARDS, PRINCESS, CYBERPUNK, ANGEL_GIRL, FAIRY_GIRL, SCENES

big_prompt_list = MAGE_WOMEN + WARRIOR_GIRL + ANGEL_GIRL + FAIRY_GIRL + PRINCESS + CYBERPUNK + DEMON_GIRL + DARK_FIGURES + NECROMANCERS + WARRIORS + DARK_WIZARDS + SCENES

# big_prompt_list = SCENES

prefix = "Photo of "
suffix = " Cinematic photograph, photo realistic, realistic skin, ultra detailed, realism, film grain, UHD, highest-quality, intricate details"

suffix_anime = "anime, anime_arts, illustration, line art, anime shading, very aesthetic, high quality"

magic_of_art = "<lora:Magic of Art (FLUX):0.5>"
magic_of_art_triggers = "magic style"


mythport = "mythp0rt, <lora:FluxMythP0rtr4itStyle:0.3>"

velmysticrealism = "R3alisticF, <lora:FluxMythR3alisticF:0.6> "

movie_portrait = "Movie_Portrait, <lora:Movie_Portrait:0.4>"

cinematic_style = "<lora:Cinematic style 3 (FLUX):0.25>"

mysticFantasy = "<lora:MysticFantasy:0.8>"
mysticRealism = "<lora:MysticRealism:0.25>"

sxzdarkFantasy = "drkfnts style, <lora:sxz-Dark-Fantasy-v2-Flux:0.4>"

realism = "Realism, <lora:Realism2:0.1>"

elden = "elden ring style, dark atmosphere, chromatic aberration, <lora:sxz-eldenring-aitoolkit-flux:0.4>"

chadmward= "drk4rt, <lora:ChadMichaelWard:0.8>"

aidmaImageUprader = "aidmaimageupgrader, <lora:aidmaImageUprader-FLUX-v03:0.4>"
aidmaImageUpgraderPro = "aidmafluxpro1.1, <lora:aidmaFLUXPro1.1-FLUX-v0.3:0.2>"
aidmarealisticskin = "aidmarealisticskin, <lora:aidmaRealisticSkin-FLUX-v01:0.4>"
aidmaphoto = "<lora:Eldritch_Photography_for_Flux_133:0.1>"

warcraft = "<lora:sxz-Warcraft-Cinematic-Flux:0.8>, wrcrftcnmtc"

hands_lora = "<lora:FluxDetailed_Hands3:0.8>"
hand = "perfect detailed gloved hand, perfection "

ultra_real_photo = "<lora:FluxUltraRealPhoto:0.2>"

abstract_arts = "abstract_arts, <lora:abstract_arts_flux_lora_03:1.8>"

RMfantasify = "<lora:RM_Fantastify_v0.8M_classic:0.8>"

mistoon_anime = "<lora:Mistoon_Anime_Flux:0.2>"
MJanime = "<lora:MJanime_Flux_LoRa_v3_Final:0.8>"

flat_color_anime  = "flat colour anime style image showing, <lora:flat_colour_anime_style_v3.4:0.5>"
fca_style = "fca_style, <lora:fca_style_v3.3:0.2>"
fc_anime = "<lora:fcanime_v1:0.5>"
anime_flucifer= "<lora:Anime Mystery Style _ Flux _ Lucifer_v1.0:0.>"
anime_era = "Anime_Era, <lora:Anime_Era_Flux_Lora:0.2>"
anime_art = "<lora:Anime v1.3:0.2>"
animefy = "<lora:RM_Animefy_v1.0M:0.95>"
fsstyle = "fsstyle, <lora:FluxFsstyle:0.2>"
cyberpunk_anime = "<lora:CPA:0.2>"
anime_lines = "An1meL1nes, <lora:FluxMythAn1meL1nes:0.25>"

painterly = "in the style of ckpf, <lora:ck-painterly-fantasy-flux:0.8>"
gothic_lines = "G0thicL1nes, <lora:FluxMythG0thicL1nes:0.5>"
mezzomint = "Mezzotint_Artstyle, <lora:Mezzotint_Artstyle_for_Flux:0.5>"

bow = "bow and arrow, <lora:bowandarrowv1:1.0>"

# reedy, nistyle <lora:aidmaImageUpgrader-FLUX-V0.2:0.8> <lora:aidmaMJ6.1_aidmaMJ6.1-FLUX-v0.4:0.8> <lora:reedy-art-style:0.7>

negative_prompt = ""

def wrap(
    model_name,
    prompt,
    negative_prompt="",
    hand_prompt="",
    steps=35,
    ):

    if not hand_prompt:
        hand_prompt = prompt

    t2i_adetailer_face_hand_workflow(
        model_name,
        prompt,
        negative_prompt,
        prompt,
        hand_prompt,
        steps=steps,
        width=1152,
        height=1408,
        cfg_scale=1,
        distilled_cfg_scale=4,
        # sampler_name="[Forge] Flux Realistic (2x Slow)",
        # sampler_name="[Forge] Flux Realistic",
        # sampler_name="DEIS",
        sampler_name="Euler",
        # sampler_name="DPM++ 2M",
        scheduler="Beta",
        ad_inpaint_width=1024,
        ad_inpaint_height=1024,
        ad_use_steps=True,
        ad_steps=64,
    )

webui_server_url = WEBUI_SERVER_URL
IMAGES_PER_MODEL = 4
t2i_model_name = "lyhAnime_korIl01"
# project0_v40ArtRealismFP8
# jibMixFlux_v7PixelHeavenBeta
# jibMixFlux_v8Accentueight
# t2i_model_name = 'jibMixFlux_v61RealPixFixed'
change_model(webui_server_url, t2i_model_name)

# big_prompt_list = []
# with jsonlines.open('outputs.jsonl') as reader:
#     for obj in reader:
#         big_prompt_list.append(obj['prompt'])

for prompt in big_prompt_list:
    # prompt_mod = f"{prompt},{suffix}"
    # prompt_mod = f"{prefix}{prompt},{suffix}"
    # prompt_mod = f"{prompt},{suffix},{magic_of_art_triggers},{magic_of_art}"
    # prompt_mod = f"{prompt},{suffix},{aidmaImageUprader}"
    # prompt_mod = f"{prompt},{mythport},{suffix}"
    # prompt_mod = f"{prompt},{warcraft},{velmysticrealism},{mythport},{suffix}"
    # prompt_mod = f"{prompt},{velmysticrealism},{suffix}"
    # prompt_mod = f"{prompt},{mysticRealism},{aidmaImageUpgraderPro},{suffix}"
    # prompt_mod = f"{prompt},{aidmaImageUpgraderPro},{suffix}"
    # prompt_mod = f"{prompt},{warcraft},{aidmaImageUpgraderPro},{suffix}"
    # prompt_mod = f"{prompt},{warcraft},{aidmaImageUpgraderPro},{mythport},{suffix}"
    # prompt_mod = f"{prompt},{aidmaImageUpgraderPro},{mythport},{suffix_anime}"
    # prompt_mod = f"{prompt},{aidmaImageUpgraderPro},{velmysticrealism},{suffix}"
    # prompt_mod = f"{prompt},{velmysticrealism},{mythport},{suffix}"
    # prompt_mod = f"{suffix},{prompt},{aidmarealisticskin},{aidmaphoto},{mysticRealism},{mythport}"
    # prompt_mod = f"{suffix},{prompt},{aidmarealisticskin},{aidmaphoto},{mysticRealism}"
    # prompt_mod = f"{suffix},{prompt},{aidmarealisticskin},{mysticRealism}"
    # prompt_mod = f"{suffix},{prompt},{velmysticrealism},{mysticRealism},{movie_portrait},{mythport},{aidmaImageUpgraderPro}"
    # prompt_mod = f"{prompt},{suffix},{aidmarealisticskin},{mysticRealism},{cinematic_style},{mythport},{aidmaImageUpgraderPro}"
    # prompt_mod = f"{prompt},{suffix},{aidmarealisticskin},{mysticRealism},{cinematic_style},{mythport}"
    # prompt_mod = f"{prompt},{mythport},{aidmarealisticskin},{suffix}"

    # prompt_mod = f"{prompt},{suffix_anime},{animefy}"
    # prompt_mod = f"{prompt},{suffix_anime},{anime_art}"
    # prompt_mod = f"{prompt},{suffix_anime},{anime_era}"
    # prompt_mod = f"{prompt},{suffix_anime},{fc_anime}"
    # prompt_mod = f"{prompt},{suffix_anime},{anime_flucifer}"
    # prompt_mod = f"{prompt},{suffix_anime},{fc_anime},{animefy}, {anime_flucifer}"
    # prompt_mod = f"{prompt},{suffix_anime},{fc_anime},{animefy},{anime_art}"
    # prompt_mod = f"{prompt},{suffix_anime},{fc_anime},{animefy},{cyberpunk_anime}"
    prompt_mod = f"{prompt},{suffix_anime},{animefy},{fc_anime},{anime_art},{cyberpunk_anime},{fsstyle},{anime_era}"
    # prompt_mod = f"{prompt},{suffix_anime},{animefy},{fc_anime},{anime_art},{cyberpunk_anime},{anime_flucifer},{mistoon_anime}"
    # prompt_mod = f"{prompt},{suffix_anime},{fc_anime},{animefy},{anime_art},{cyberpunk_anime},{anime_flucifer},{anime_lines}"
    # prompt_mod = f"{prompt},{suffix_anime},{animefy},{anime_art},{anime_era},{fc_anime}"
    # prompt_mod = f"{prompt},{suffix_anime},{animefy},{anime_art},{anime_era},{fc_anime},{anime_flucifer}"
    # prompt_mod = f"{abstract_art},{prompt},{aidmaImageUpgraderPro},{suffix}"

    hand_prompt = f"{hand},{hands_lora},{suffix}"

    for i in range(4):
        wrap('',
             prompt_mod,
             hand_prompt=hand_prompt,
             steps=40
             )

print("Job Completed Successfully...")
