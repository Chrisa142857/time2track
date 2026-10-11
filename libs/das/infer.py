import os
import sys
import argparse
from PIL import Image

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from moviepy.editor import VideoFileClip
from diffusers.utils import load_image, load_video

from models.pipelines import DiffusionAsShaderPipeline, CameraMotionGenerator, ObjectMotionGenerator

def load_media(media_path, max_frames=49, transform=None):
    """Load video or image frames and convert to tensor
    
    Args:
        media_path (str): Path to video or image file
        max_frames (int): Maximum number of frames to load
        transform (callable): Transform to apply to frames
        
    Returns:
        Tuple[torch.Tensor, float]: Video tensor [T,C,H,W] and FPS
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((480, 720)),
            transforms.ToTensor()
        ])
    
    # Determine if input is video or image based on extension
    ext = os.path.splitext(media_path)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov']
    
    if is_video:
        frames = load_video(media_path)
        fps = len(frames) / VideoFileClip(media_path).duration
    else:
        # Handle image as single frame
        image = load_image(media_path)
        frames = [image]
        fps = 8  # Default fps for images
    
    # Ensure we have exactly max_frames
    if len(frames) > max_frames:
        frames = frames[:max_frames]
    elif len(frames) < max_frames:
        last_frame = frames[-1]
        while len(frames) < max_frames:
            frames.append(last_frame.copy())
            
    # Convert frames to tensor
    video_tensor = torch.stack([transform(frame) for frame in frames])
    
    return video_tensor, fps, is_video

def inference(das, prompt, checkpoint_path, tracking_path, output_dir, input_path):
    video_tensor, fps, is_video = load_media(input_path)
    tracking_tensor, _, _ = load_media(args.tracking_path)
    das.apply_tracking(
        video_tensor=video_tensor,
        fps=8,
        tracking_tensor=tracking_tensor,
        img_cond_tensor=None,
        prompt=prompt,
        checkpoint_path=checkpoint_path
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None, help='Path to input video/image')
    parser.add_argument('--prompt', type=str, required=True, help='Repaint prompt')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--depth_path', type=str, default=None, help='Path to depth image')
    parser.add_argument('--tracking_path', type=str, default=None, help='Path to tracking video, if provided, camera motion and object manipulation will not be applied')
    parser.add_argument('--repaint', type=str, default=None, 
                       help='Path to repainted image, or "true" to perform repainting, if not provided use original frame')
    parser.add_argument('--camera_motion', type=str, default=None, help='Camera motion mode')
    parser.add_argument('--object_motion', type=str, default=None, help='Object motion mode: up/down/left/right')
    parser.add_argument('--object_mask', type=str, default=None, help='Path to object mask image (binary image)')
    parser.add_argument('--tracking_method', type=str, default="spatracker", 
                        help='default tracking method for image input: moge/spatracker, if \'moge\' method will extract first frame for video input')
    parser.add_argument('--coarse_video_path', type=str, default=None, help='Path to coarse video for object motion')
    parser.add_argument('--start_noise_t', type=int, default=10, help='Strength of object motion')
    args = parser.parse_args()
    
    # Load input video/image
    video_tensor, fps, is_video = load_media(args.input_path)

    # Initialize pipeline
    das = DiffusionAsShaderPipeline(gpu_id=args.gpu, output_dir=args.output_dir)
    
    # Repaint first frame if requested
    repaint_img_tensor = None

    # Generate tracking if not provided
    tracking_tensor = None
    pred_tracks = None
    cam_motion = CameraMotionGenerator(args.camera_motion)

    if args.tracking_path:
        tracking_tensor, _, _ = load_media(args.tracking_path)
    
    coarse_video = load_media(args.coarse_video_path)[0] if args.coarse_video_path else None
    
    das.apply_tracking(
        video_tensor=video_tensor,
        fps=24,
        tracking_tensor=tracking_tensor,
        img_cond_tensor=repaint_img_tensor,
        prompt=args.prompt,
        checkpoint_path=args.checkpoint_path,
        coarse_video=coarse_video,
        start_noise_t=args.start_noise_t
    )
