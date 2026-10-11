import os
import argparse
import rembg
import numpy as np
import math
import torch
import json

from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from diffusers_sv3d import SV3DUNetSpatioTemporalConditionModel, StableVideo3DDiffusionPipeline
from kiui.cam import orbit_camera

SV3D_DIFFUSERS = "chenguolin/sv3d-diffusers"

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "~/.cache/huggingface"

def construct_camera(azimuths_rad, elevation_rad, output_dir, res=576, radius=2, fov=33.8):
    
    transforms = {}
    transforms["camera_angle_x"] = math.radians(fov)
    transforms["frames"] = []
    
    for i in range(21):
        frame = {}
        frame['file_path'] = f"data/{i:03d}"
        frame['transform_matrix'] = orbit_camera(elevation_rad[i], azimuths_rad[i], radius, is_degree=False).tolist()
        transforms['frames'].append(frame)
    
    with open(f"{output_dir}/../transforms_train.json", "w") as f:
        json.dump(transforms, f, indent=4)
    with open(f"{output_dir}/../transforms_val.json", "w") as f:
        json.dump(transforms, f, indent=4)
    with open(f"{output_dir}/../transforms_test.json", "w") as f:
        json.dump(transforms, f, indent=4)

def recenter(image, h_begin=100, w_begin=220, res=256):
    image = np.array(image)
    h_image, w_image = image.shape[:2]
    new_image = np.zeros((res, res, 4), dtype=np.uint8) 
    h_begin_new = -min(0, h_begin)
    w_begin_new = -min(0, w_begin)
    if h_begin > 0 and w_begin > 0:
        new_image = image[h_begin:h_begin+res, w_begin:w_begin+res]
    else:
        new_image[h_begin_new:h_begin_new+h_image, w_begin_new:w_image] = image
    return Image.fromarray(new_image)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="../../data", type=str, help="Base dir")
    parser.add_argument("--output-dir", default="../../data", type=str, help="Output filepath")
    parser.add_argument("--data-name", default="chair", type=str, help="Data Name")
    parser.add_argument("--elevation", default=0, type=float, help="Camera elevation of the input image")
    parser.add_argument("--half-precision", action="store_true", help="Use fp16 half precision")
    parser.add_argument("--seed", default=-1, type=int, help="Random seed")
    args = parser.parse_args()

    image_path = f'{args.base_dir}/{args.data_name}/{args.data_name}.png'
    output_dir = f'{args.output_dir}/{args.data_name}/data'
    os.makedirs(output_dir, exist_ok=True)

    num_frames, sv3d_res = 20, 576
    elevations_deg = [args.elevation] * num_frames
    elevations_rad = [np.deg2rad(e) for e in elevations_deg]
    polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
    azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
    azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    azimuths_rad[:-1].sort()
    
    # print(f"Elevation: {elevations_rad}")
    print(f"Azimuth: {np.rad2deg(azimuths_rad)}")
    # construct_camera(azimuths_rad, elevations_rad, output_dir=output_dir)
    
    bg_remover = rembg.new_session()
    unet = SV3DUNetSpatioTemporalConditionModel.from_pretrained(SV3D_DIFFUSERS, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(SV3D_DIFFUSERS, subfolder="vae")
    scheduler = EulerDiscreteScheduler.from_pretrained(SV3D_DIFFUSERS, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(SV3D_DIFFUSERS, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(SV3D_DIFFUSERS, subfolder="feature_extractor")

    pipeline = StableVideo3DDiffusionPipeline(
        image_encoder=image_encoder, feature_extractor=feature_extractor, 
        unet=unet, vae=vae,
        scheduler=scheduler,
    )
    pipeline = pipeline.to("cuda")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16 if args.half_precision else torch.float32, enabled=True):
                
            h_begin, w_begin, res = 180, 190, 280
            image = Image.open(image_path)
            image = recenter(image, h_begin, w_begin, res)
            image = rembg.remove(image, session=bg_remover) # [H, W, 4]
            image.save(f"{output_dir}/../{args.data_name}_alpha.png")
            if len(image.split()) == 4:  # RGBA
                input_image = Image.new("RGB", image.size, (255, 255, 255))  # pure white bg
                input_image.paste(image, mask=image.split()[3])  # 3rd is the alpha channel
            else:
                input_image = image
            
            video_frames = pipeline(
                input_image.resize((sv3d_res, sv3d_res)),
                height=sv3d_res,
                width=sv3d_res,
                num_frames=num_frames,
                decode_chunk_size=8,  # smaller to save memory
                polars_rad=polars_rad,
                azimuths_rad=azimuths_rad,
                generator=torch.manual_seed(args.seed) if args.seed >= 0 else None,
            ).frames[0]

    os.makedirs(output_dir, exist_ok=True)
    export_to_gif(video_frames, f"{output_dir}/animation.gif", fps=7)
    for i, frame in enumerate(video_frames):
        # frame = frame.resize((res, res))
        frame.save(f"{output_dir}/{i:03d}.png")
    video_frames[19].save(f"../LGM/workspace_test/{args.data_name}_0.png")
    video_frames[4].save(f"../LGM/workspace_test/{args.data_name}_1.png")
    video_frames[9].save(f"../LGM/workspace_test/{args.data_name}_2.png")
    video_frames[14].save(f"../LGM/workspace_test/{args.data_name}_3.png")
    
    
if __name__ == "__main__":
    main()
    