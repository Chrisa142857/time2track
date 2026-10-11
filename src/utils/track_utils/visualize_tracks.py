import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from .visualizer import Visualizer

def create_white_video(num_frames, target_h=480, target_w=720):
    white_video = torch.ones((1, num_frames, 3, target_h, target_w))
    return white_video

def process_video(tracks_path, output_dir, args):

    video_name = os.path.splitext(os.path.basename(tracks_path))[0].replace('_tracks', '')
    video = create_white_video(args.num_frames)

    combined_data = np.load(tracks_path, allow_pickle=True).item()
    tracks = torch.from_numpy(combined_data['tracks'])
    visibility = torch.from_numpy(combined_data['visibility'])

    vis = Visualizer(
        save_dir=output_dir,
        grayscale=False,
        fps=args.output_fps,
        pad_value=0,
        linewidth=args.point_size,
        tracks_leave_trace=args.len_track
    )
    
    video_vis = vis.visualize(
        video=video,
        tracks=tracks,
        visibility=visibility,
        filename=video_name
    ) 

def visualize_tracks(tracks_dir, output_dir, args):
    
    args.tracks_dir = tracks_dir

    os.makedirs(output_dir, exist_ok=True)
    tracks_files = [f for f in os.listdir(args.tracks_dir) if f.endswith('tracks.npy')]
    for tracks_file in tracks_files:
        tracks_path = os.path.join(args.tracks_dir, tracks_file)
        process_video(tracks_path, output_dir, args)