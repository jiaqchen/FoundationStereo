# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os, sys
import glob
import re
import numpy as np
import cv2
import imageio
import torch
import logging
from pathlib import Path
from tqdm import tqdm
# import rerun as rr

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *

def find_stereo_pairs(folder_path):
    """Find matching left/right image pairs in the folder."""
    left_files = sorted(glob.glob(os.path.join(folder_path, "left_*.png")))
    right_files = sorted(glob.glob(os.path.join(folder_path, "right_*.png")))
    
    pairs = []
    for left_file in left_files:
        # Extract the number from left_XXXX.png
        left_name = os.path.basename(left_file)
        match = re.search(r'left_(\d+)\.png', left_name)
        if match:
            number = match.group(1)
            right_file = os.path.join(folder_path, f"right_{number}.png")
            if os.path.exists(right_file):
                pairs.append((left_file, right_file))
            else:
                logging.warning(f"No matching right image found for {left_file}")
    
    logging.info(f"Found {len(pairs)} stereo pairs")
    return pairs

def process_stereo_pair(model, left_file, right_file, args, frame_idx):
    """Process a single stereo pair and return depth map and point cloud data."""
    img0 = imageio.imread(left_file)
    img1 = imageio.imread(right_file)
    
    scale = args.scale
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    H, W = img0.shape[:2]
    img0_ori = img0.copy()
    
    img0_tensor = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1_tensor = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0_tensor.shape, divis_by=32, force_square=False)
    img0_tensor, img1_tensor = padder.pad(img0_tensor, img1_tensor)
    
    with torch.cuda.amp.autocast(True):
        if not args.hiera:
            disp = model.forward(img0_tensor, img1_tensor, iters=args.valid_iters, test_mode=True)
        else:
            disp = model.run_hierachical(img0_tensor, img1_tensor, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)
    
    # Remove invisible points
    if args.remove_invisible:
        yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
        us_right = xx - disp
        invalid = us_right < 0
        disp[invalid] = np.inf
    
    # Convert disparity to depth
    with open(args.intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
        baseline = float(lines[1])
    K[:2] *= scale
    depth = K[0,0] * baseline / disp
    
    # Generate point cloud data
    xyz_map = depth2xyzmap(depth, K)
    
    # Filter points
    valid_mask = (depth > 0) & (depth <= args.z_far) & ~np.isinf(depth)
    
    points = xyz_map[valid_mask]
    colors = img0_ori[valid_mask]
    
    return depth, points, colors, img0_ori

def create_depth_video(depth_maps, output_path, fps=10):
    """Create a video from depth maps."""
    if not depth_maps:
        logging.warning("No depth maps to create video")
        return
    
    # Normalize depth maps for visualization
    all_depths = np.concatenate([d[~np.isinf(d)].flatten() for d in depth_maps])
    vmin, vmax = np.percentile(all_depths, [5, 95])
    
    H, W = depth_maps[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    for depth in depth_maps:
        # Normalize and convert to 8-bit
        depth_vis = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
        depth_vis[np.isinf(depth)] = 0  # Set invalid depths to black
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        out.write(depth_colored)
    
    out.release()
    logging.info(f"Depth video saved to {output_path}")

def visualize_with_rerun(point_cloud_data, output_dir):
    """Visualize point clouds over time using rerun.io."""
    rr.init("stereo_depth_sequence", spawn=True)
    
    for frame_idx, (points, colors) in enumerate(point_cloud_data):
        if len(points) == 0:
            continue
            
        # Log the point cloud for this frame
        rr.set_time_sequence("frame", frame_idx)
        
        # Convert colors to 0-1 range if they're in 0-255 range
        if colors.max() > 1.0:
            colors = colors.astype(np.float32) / 255.0
        
        rr.log(
            "world/points",
            rr.Points3D(
                positions=points,
                colors=colors,
                radii=0.01
            )
        )
        
        # Also log some metadata
        rr.log("metadata/frame_info", rr.TextLog(f"Frame {frame_idx}: {len(points)} points"))
    
    logging.info("Rerun visualization started. Check the rerun viewer.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required=True, type=str, help='Folder containing left_XXXX.png and right_XXXX.png files')
    parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
    parser.add_argument('--out_dir', default=f'{code_dir}/output_folder/', type=str, help='the directory to save results')
    parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud')
    parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
    parser.add_argument('--fps', type=int, default=10, help='frames per second for depth video')
    parser.add_argument('--save_individual_clouds', type=int, default=0, help='save individual point cloud files')
    args = parser.parse_args()
    
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load model
    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")
    
    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_dir, weights_only=False)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()
    
    # Find stereo pairs
    stereo_pairs = find_stereo_pairs(args.input_folder)
    if not stereo_pairs:
        logging.error("No stereo pairs found in the input folder")
        sys.exit(1)
    
    # Process all pairs
    depth_maps = []
    point_cloud_data = []
    
    logging.info("Processing stereo pairs...")
    for i, (left_file, right_file) in enumerate(tqdm(stereo_pairs)):
        logging.info(f"Processing pair {i+1}/{len(stereo_pairs)}: {os.path.basename(left_file)}, {os.path.basename(right_file)}")
        
        try:
            depth, points, colors, img_ori = process_stereo_pair(model, left_file, right_file, args, i)
            depth_maps.append(depth)
            point_cloud_data.append((points, colors))
            
            # Save depth map
            np.save(f'{args.out_dir}/depth_{i:04d}.npy', depth)
            
            # Optionally save individual point clouds
            if args.save_individual_clouds:
                if len(points) > 0:
                    pcd = toOpen3dCloud(points, colors)
                    if args.denoise_cloud and len(points) > args.denoise_nb_points:
                        cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
                        pcd = pcd.select_by_index(ind)
                    o3d.io.write_point_cloud(f'{args.out_dir}/cloud_{i:04d}.ply', pcd)
            
            # Save visualization image
            vis = vis_disparity(depth * args.scale / (depth.max() if depth.max() > 0 else 1))
            vis = np.concatenate([img_ori, vis], axis=1)
            imageio.imwrite(f'{args.out_dir}/vis_{i:04d}.png', vis)
            
        except Exception as e:
            logging.error(f"Error processing pair {i}: {e}")
            continue
    
    # Create depth video
    if depth_maps:
        video_path = f'{args.out_dir}/depth_sequence.mp4'
        create_depth_video(depth_maps, video_path, args.fps)
    
    # Visualize with rerun
    # if point_cloud_data:
    #     logging.info("Starting rerun visualization...")
    #     visualize_with_rerun(point_cloud_data, args.out_dir)
    
    logging.info(f"Processing complete. Results saved to {args.out_dir}")
    logging.info(f"Generated {len(depth_maps)} depth maps and point clouds")
