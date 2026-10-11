import numpy as np
import h5py
import torch
import torch.nn.functional as F
import random
import json
from tqdm import tqdm

def chamfer_distance(points_pred, points_gt):
    x, y = points_pred, points_gt
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(points_pred).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return (P.min(1)[0].mean(dim=1) + P.min(2)[0].mean(dim=1))

def chamfer_distance1(pc1, pc2):
    # pairwise_dist: (N1, N2)
    pairwise_dist = torch.cdist(pc1, pc2, p=2) ** 2 # Euclidean distances
    
    min_dist_pc1_to_pc2 = pairwise_dist.min(dim=1)[0]  # shape (N1,)
    min_dist_pc2_to_pc1 = pairwise_dist.min(dim=0)[0]  # shape (N2,)
    
    chamfer = min_dist_pc1_to_pc2.mean() + min_dist_pc2_to_pc1.mean()
    return chamfer

def points_to_voxel_grid(points, voxel_size, grid_min, grid_max):
    """Converts a point cloud to a voxel grid representation.

    Args:
        points (torch.Tensor): (N, 3) tensor of point coordinates.
        voxel_size (float): Size of each voxel.
        grid_min (torch.Tensor): (3,) tensor of minimum grid coordinates.
        grid_max (torch.Tensor): (3,) tensor of maximum grid coordinates.

    Returns:
        torch.Tensor: (Dx, Dy, Dz) boolean tensor representing the voxel grid.
    """
    device = points.device
    grid_size = ((grid_max - grid_min) / voxel_size).round().int()
    grid = torch.zeros(*grid_size, dtype=torch.bool, device=device)

    valid_indices = (
        (points >= grid_min).all(dim=1) & (points < grid_max).all(dim=1)
    )
    valid_points = points[valid_indices]
    voxel_indices = ((valid_points - grid_min) / voxel_size).floor().long()
    # print(grid_size, voxel_indices.max(), valid_points.max())
    # import pdb; pdb.set_trace();

    grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True
    return grid

def volume_iou(grid1, grid2):
    """Calculates the Volume IoU between two voxel grids.
    Args:
        grid1 (torch.Tensor): (Dx, Dy, Dz) boolean tensor representing the first voxel grid.
        grid2 (torch.Tensor): (Dx, Dy, Dz) boolean tensor representing the second voxel grid.
    Returns:
        float: Volume IoU score.
    """
    intersection = (grid1 & grid2).sum().float()
    union = (grid1 | grid2).sum().float()
    if union == 0:
        return 0.0
    return intersection / union

def evaluate_4d(pts_pred, pts_gt):
    N = pts_pred.shape[0]
    total_iou, total_cd = 0.0, 0.0

    voxel_size = 0.1
    grid_min = torch.tensor([-1.5, -1.5, -1.5])
    grid_max = torch.tensor([1.5, 1.5, 1.5])

    for i in range(N):
        cd = chamfer_distance1(pts_pred[i], pts_gt[i])
        total_cd += cd

        grid1 = points_to_voxel_grid(pts_gt[i].reshape(-1, 3), voxel_size, grid_min, grid_max)
        grid2 = points_to_voxel_grid(pts_pred[i].reshape(-1, 3), voxel_size, grid_min, grid_max)
        iou = volume_iou(grid1, grid2)
        total_iou += iou
    total_iou /= N
    total_cd /= N
    total_mse = F.mse_loss(pts_pred, pts_gt)
    return total_iou, total_cd, total_mse

def evaluate_test_4d(pts_pred_all, pts_gt_all):
    N = pts_pred_all.shape[0]
    print(f"Num of testing samples: {N}")
    total_iou, total_cd, total_mse = 0., 0., 0.
    single_iou, single_cd, single_mse = [], [], []
    for i in tqdm(range(N)):
        seq_iou, seq_cd, seq_mse = evaluate_4d(pts_pred_all[i], pts_gt_all[i])
        single_iou.append(seq_iou)
        single_cd.append(seq_cd)
        single_mse.append(seq_mse)
        total_iou += seq_iou
        total_cd += seq_cd
        total_mse += seq_mse
    return total_iou / N, total_cd / N, total_mse / N, single_iou, single_cd, single_mse

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_lst', type=str, default='./outputs_v3_test_list.json')
    parser.add_argument('--pred_path', type=str, default='./outputs/dit_8layers_2048p_hf-objaverse-v1_24frames_pointembed_latent256_deform0.001_8gpus_v5/test_100_25steps')
    parser.add_argument('--save_html', action='store_true')
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()

    # split_lst = json.load(open('/mnt/kostas-graid/datasets/chenwang/traj/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/hf-objaverse-v1_valid_sim_list_v3_test.json'))
    split_lst = json.load(open(args.split_lst))
    random.seed(0)
    random.shuffle(split_lst)
    split_lst = split_lst[:100]
    print(split_lst)

    pts_gt_all = []
    pts_pred_all = []
    gif_paths = []

    import glob, os
    for i in range(len(split_lst)):
        model_metas = h5py.File(f'/mnt/kostas-graid/datasets/chenwang/traj/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/outputs_v3/{split_lst[i]}')
        pts_gt = np.array(model_metas['x'])[1:48:2]
        pts_gt = (pts_gt - 5) / 2
        pts_pred = np.load(f"{args.pred_path}/{i}_{torch.log10(torch.from_numpy(np.array(model_metas['E'])).float()):03f}_{np.array(model_metas['nu']):03f}.npy")

        pts_gt_all.append(pts_gt)
        pts_pred_all.append(pts_pred)
    pts_gt_all = np.stack(pts_gt_all, axis=0)
    pts_pred_all = np.stack(pts_pred_all, axis=0)
    print("Loaded all test samples")
    iou, cd, mse, single_iou, single_cd, single_mse = evaluate_test_4d(torch.from_numpy(pts_gt_all), torch.from_numpy(pts_pred_all))
    print("IOU, CD, MSE:", iou, cd, mse)

    if args.save_html:
        sorted_indices = np.argsort(single_iou)
        pts_gt_all = pts_gt_all[sorted_indices]
        pts_pred_all = pts_pred_all[sorted_indices]
        single_iou = np.array(single_iou)[sorted_indices]
        single_cd = np.array(single_cd)[sorted_indices]
        single_mse = np.array(single_mse)[sorted_indices]
        gif_paths = [gif_paths[i] for i in sorted_indices]

        import html
        rows = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "  <meta charset='utf-8'>",
            "  <title>GIF gallery</title>",
            "  <style>",
            "     body{margin:0;font-family:sans-serif;background:#fafafa;color:#333}",
            "     .row{padding:16px;text-align:center;border-bottom:1px solid #eee;}",
            "     img{max-width:50%;height:auto;display:block;margin:0 auto;}",
            "     .caption{margin-top:8px;font-size:0.9rem;word-break:break-all;}",
            "  </style>",
            "</head>",
            "<body>",
        ]

        # 4) one <div> per gif with caption
        for i, gif in enumerate(gif_paths):
            name = gif                      # full file name (incl. .gif)
            alt  = html.escape(gif)         # alt text sans extension
            rows.append(
                f"  <div class='row'>"
                f"<img src='{name.split('/')[-1]}' alt='{alt}'>"
                f"<p class='caption'>"
                f"Name: {html.escape(name)}<br>"
                f"IoU: {single_iou[i]:.4f}, Chamfer Distance: {single_cd[i]:.4f}, MSE: {single_mse[i]:.4f}"
                f"</p>"
                f"</div>"
            )

        rows += ["</body>", "</html>"]
        with open('./outputs/dit_8layers_2048p_hf-objaverse-v1_24frames_pointembed_latent256_deform0.001_8gpus_v3_all/test_1000/visualize.html', 'w') as f:
            f.write('\n'.join(rows))