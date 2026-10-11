import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import json

from PIL import Image
from io import BytesIO
import sys, pathlib, html

def camera_view_dir_y(elev, azim):
    """Unit vector for camera direction with Y as 'up'."""
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    dx = np.sin(azim_rad) * np.cos(elev_rad)
    dy = np.sin(elev_rad)
    dz = np.cos(azim_rad) * np.cos(elev_rad)
    return np.array([dx, dy, dz])

def compute_depth(points, elev, azim):
    """Project points onto the camera's view direction (Y as 'up')."""
    view_dir = camera_view_dir_y(elev, azim)
    # depth = p · view_dir
    depth = points @ view_dir
    return depth

## Should merge with the following function, but no time for now.
def save_pointcloud_video(points_pred, points_gt, save_path, drag_mask=None, fps=48, point_color='blue', vis_flag=''):
    # Configure the figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    
    if 'heartmesh' in vis_flag:
        # Bounds derived from data (centered mesh, scale may vary)
        all_pts = np.concatenate([
            points_pred.reshape(-1, 3), points_gt.reshape(-1, 3)
        ], axis=0)
        pad = 0.05 * (all_pts.max() - all_pts.min()) + 1e-6
        x_min, x_max = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
        y_min, y_max = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
        z_min, z_max = all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad
        elev, azim = 20, 45   # anterior-superior view for cardiac anatomy
    elif 'objaverse' in vis_flag:
        x_max, y_max, z_max = 1.5, 1.5, 1.5
        x_min, y_min, z_min = -1.5, -1.5, -1.5
        elev, azim = 45, 225
    else:
        x_max, y_max, z_max = 1, 1, 1
        x_min, y_min, z_min = -1, -1, -1
        elev, azim = 45, 225   # default; also used for shapenet

    ax.view_init(elev=elev, azim=azim, vertical_axis='y')
        
    # Plot and save each frame
    cmap_1 = plt.colormaps.get_cmap('cool')
    cmap_2 = plt.colormaps.get_cmap('autumn')
    frames_pred = []
    frames_gt = []

    if drag_mask is not None and drag_mask.sum() == 0:
        drag_mask = None
    
    for label, points in [('pred', points_pred), ('gt', points_gt)]:
        
        for i in range(points.shape[0]):
            
            frame_points = points[i]
            if drag_mask is not None and not (drag_mask == True).all():
                drag_mask = (drag_mask == 1.0)
                drag_points = frame_points[drag_mask]
                frame_points = frame_points[~drag_mask]
                
            depth_frame_points = compute_depth(frame_points, elev=elev, azim=azim)
            depth_frame_points_normalized = (depth_frame_points - depth_frame_points.min()) / \
                (depth_frame_points.max() - depth_frame_points.min())
            color_frame_points = cmap_1(depth_frame_points_normalized)

            if drag_mask is not None and not (drag_mask == True).all():
                frame_points_drag = drag_points
                depth_frame_points_drag = compute_depth(frame_points_drag, elev=elev, azim=azim)
                depth_frame_points_drag_normalized = (depth_frame_points_drag - depth_frame_points_drag.min()) / \
                    (depth_frame_points_drag.max() - depth_frame_points_drag.min())
                color_frame_points_drag = cmap_2(np.ones_like(depth_frame_points_drag_normalized) * -10)
                all_points = np.concatenate([frame_points, frame_points_drag], axis=0)
                all_color = np.concatenate([color_frame_points, color_frame_points_drag], axis=0)
            else:
                all_points, all_color = frame_points, color_frame_points
                
            
            ax.clear()
            ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c=all_color, s=1, depthshade=False)
            
            ax.axis('off')  # Turn off the axes
            ax.grid(False)  # Hide the grid
            
            # Set equal aspect ratio
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            
            # Adjust margins for tight layout
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            # Save frame
            buf = BytesIO()
            plt.savefig(buf, bbox_inches='tight', pad_inches=0.0, dpi=300)
            buf.seek(0)
            
            if label == 'pred':
                frames_pred.append(Image.open(buf))
            else:
                frames_gt.append(Image.open(buf))
                
    plt.close()
    frames = []
    for i in range(len(frames_pred)):
        frame = np.concatenate([np.array(frames_pred[i]), np.array(frames_gt[i])], axis=1)
        frames.append(Image.fromarray(frame))
    frames[0].save(save_path, save_all=True, append_images=frames[1:], fps=fps, loop=0)

def save_mesh_video(points_pred, points_gt, faces_tri, save_path, fps=48, vis_flag=''):
    """Render predicted and GT mesh trajectories side-by-side as a GIF.

    Parameters
    ----------
    points_pred, points_gt : (T, N, 3) np.ndarray
    faces_tri : (F, 3) np.ndarray of int – triangle face indices
    save_path : str – output .gif path
    """
    fig, (ax_pred, ax_gt) = plt.subplots(1, 2, figsize=(12, 6),
                                          subplot_kw={'projection': '3d'})
    for ax in (ax_pred, ax_gt):
        ax.set_box_aspect([1, 1, 1])

    all_pts = np.concatenate([
        points_pred.reshape(-1, 3), points_gt.reshape(-1, 3)
    ], axis=0)
    pad = 0.05 * (all_pts.max() - all_pts.min()) + 1e-6
    x_min, x_max = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    y_min, y_max = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
    z_min, z_max = all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad
    elev, azim = 20, 45

    face_alpha = 0.3
    edge_alpha = 0.4
    cmap_pred = plt.colormaps.get_cmap('cool')
    cmap_gt = plt.colormaps.get_cmap('autumn')

    frames = []

    for i in range(points_pred.shape[0]):
        for ax, points, cmap, title in [
            (ax_pred, points_pred, cmap_pred, 'Prediction'),
            (ax_gt, points_gt, cmap_gt, 'Ground Truth'),
        ]:
            verts = points[i]  # (N, 3)

            face_centers = verts[faces_tri].mean(axis=1)  # (F, 3)
            depth = compute_depth(face_centers, elev=elev, azim=azim)
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            face_colors = cmap(depth_norm)
            face_colors[:, 3] = face_alpha

            tri_verts = verts[faces_tri]  # (F, 3, 3)

            ax.clear()
            mesh_collection = Poly3DCollection(
                tri_verts,
                facecolors=face_colors,
                edgecolors=(0.2, 0.2, 0.2, edge_alpha),
                linewidths=0.2,
            )
            ax.add_collection3d(mesh_collection)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.view_init(elev=elev, azim=azim, vertical_axis='y')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=0)
            ax.axis('off')
            ax.grid(False)

        fig.suptitle(f'Frame {i + 1} / {points_pred.shape[0]}', fontsize=12, y=0.02)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.05, wspace=0.02)

        buf = BytesIO()
        fig.savefig(buf, bbox_inches='tight', pad_inches=0.05, dpi=200)
        buf.seek(0)
        frames.append(Image.open(buf))

    plt.close()
    frames[0].save(save_path, save_all=True, append_images=frames[1:], fps=fps, loop=0)


def save_pointcloud_video_wdp(points, drag_points, save_path, fps=48, point_color='blue', grid_lim=4, vertical_axis='y',
    elev=45, azim=225, zoom_in=False):
    
    # Configure the figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    
    if zoom_in:
        axis_min = grid_lim / 2 - 2
        axis_max = grid_lim / 2 + 2
    else:
        axis_min = grid_lim / 2 - 3
        axis_max = grid_lim / 2 + 3
    ax.view_init(elev=elev, azim=azim, vertical_axis=vertical_axis)
    
    # Plot and save each frame
    cmap_1 = plt.colormaps.get_cmap('cool')
    cmap_2 = plt.colormaps.get_cmap('autumn')
    frames = []
    
    for i in range(points.shape[0]):
        
        frame_points = points[i]
        depth_frame_points = compute_depth(frame_points, elev=elev, azim=azim)
        depth_frame_points_normalized = (depth_frame_points - depth_frame_points.min()) / \
            (depth_frame_points.max() - depth_frame_points.min())
        color_frame_points = cmap_1(depth_frame_points_normalized)
        if len(drag_points) > 0:
            frame_points_drag = drag_points[i]
            depth_frame_points_drag = compute_depth(frame_points_drag, elev=elev, azim=azim)
            depth_frame_points_drag_normalized = (depth_frame_points_drag - depth_frame_points_drag.min()) / \
                (depth_frame_points_drag.max() - depth_frame_points_drag.min())
            color_frame_points_drag = cmap_2(np.ones_like(depth_frame_points_drag_normalized) * -10)
            all_points = np.concatenate([frame_points, frame_points_drag], axis=0)
            all_color = np.concatenate([color_frame_points, color_frame_points_drag], axis=0)
        else:
            all_points, all_color = frame_points, color_frame_points
            
        ax.clear()
        ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],
            c=all_color, s=1, depthshade=False, alpha=0.9)
             
        ax.axis('off')  # Turn off the axes
        ax.grid(False)  # Hide the grid
        
        # Set equal aspect ratio
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_zlim(axis_min, axis_max)
        
        # Adjust margins for tight layout
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Save frame
        buf = BytesIO()
        plt.savefig(buf, bbox_inches='tight', pad_inches=0.0, dpi=300)
        buf.seek(0)
        frames.append(Image.open(buf))

    plt.close()
    frames[0].save(save_path, save_all=True, append_images=frames[1:], fps=fps, loop=0)

def save_pointcloud_json(points, output_json):
    """
    Generate and save a point cloud sequence to a JSON file.

    Parameters:
        num_frames (int): Number of frames in the sequence.
        num_points (int): Number of points per frame.
        output_json (str): Output JSON file path.
    """
    sequence = []
    for frame in range(points.shape[0]):
        # points = np.random.uniform(-1.5, 1.5, size=(num_points, 3)).tolist()
        sequence.append({"frame": frame, "points": points[frame].tolist()})

    # Save the sequence to a JSON file
    with open(output_json, "w") as json_file:
        json.dump({"sequence": sequence}, json_file)

def save_threejs_html(path1, path2, output_html):
    html_template = f"""
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Three.js Point Cloud Animation</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>

<body>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        const pointCloudGrids = [];

        // Load the point cloud sequence from a JSON file
        function loadSequence(filePath, offset, callback) {{
            const loader = new THREE.FileLoader();
            loader.load(filePath, function (data) {{
                const json = JSON.parse(data);
                const sequence = json.sequence;
                const pointClouds = [];

                sequence.forEach(frameData => {{
                    const points = frameData.points.map(p => new THREE.Vector3(p[0], p[1], p[2]));

                    const geometry = new THREE.BufferGeometry().setFromPoints(points);
                    const material = new THREE.PointsMaterial({{ size: 0.01, color: 0xff0000 }});

                    const pointCloud = new THREE.Points(geometry, material);

                    pointCloud.visible = false;
                    pointCloud.position.x = offset;
                    scene.add(pointCloud);

                    pointClouds.push(pointCloud);
                }});

                pointCloudGrids.push(pointClouds);
                console.log("Loaded point clouds:", filePath);
                callback();
            }}, undefined, function (err) {{
                console.error("Error loading JSON file:", err);
            }});
        }}

        camera.position.z = 5;

        let currentFrame = 0;

        function renderFrame(pointClouds, frameIndex) {{
            pointClouds.forEach(pc => pc.visible = false);

            if (pointClouds[frameIndex]) {{
                pointClouds[frameIndex].visible = true;
            }} else {{
                console.error("No point cloud for frame:", frameIndex);
            }}
        }}

        function animate() {{
            requestAnimationFrame(animate);

            controls.update();

            renderer.render(scene, camera);
        }}
        animate();

        function playSequence() {{
            setInterval(() => {{
                renderFrame(pointCloudGrids[0], currentFrame);
                renderFrame(pointCloudGrids[1], currentFrame);
                currentFrame = (currentFrame + 1) % pointCloudGrids[0].length;
            }}, 50);
        }}

        loadSequence("{path1}", -2, () => {{
            loadSequence("{path2}", 2, playSequence);
        }});
    </script>
</body>

</html>
"""
    with open(output_html, 'w') as file:
        file.write(html_template)

def vis_pcl_grid(point_clouds, save_path, grid_shape=(2, 2), bounds=[[0, 0, 0], [3, 3, 3]]):
    """
    Visualizes multiple 3D point clouds in a grid.

    Args:
        point_clouds (list of np.ndarray): A list of point clouds, each as a (N, 3) numpy array.
        grid_shape (tuple): Shape of the grid (rows, cols).
    """
    rows, cols = grid_shape
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    
    for i, point_cloud in enumerate(point_clouds):
        if i >= rows * cols:
            break  # Prevent overpopulation of the grid
        
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        ax.view_init(elev=0, azim=90)
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)

        if bounds == None:
            ax.set_xlim()
        else:
            ax.set_xlim(bounds[0][0], bounds[1][0])
            ax.set_ylim(bounds[0][1], bounds[1][1])
            ax.set_zlim(bounds[0][2], bounds[1][2])

        ax.set_title(f"Point Cloud {i + 1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
    plt.close()

def generate_html_from_exts(vis_dir, output_html, exts):
    gifs = sorted(pathlib.Path(vis_dir).glob(f'*.{exts}'))
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
    for gif in gifs:
        name = gif.name                 # full file name (incl. .gif)
        alt  = html.escape(gif.stem)    # alt text sans extension
        rows.append(
            f"  <div class='row'>"
            f"<img src='{name}' alt='{alt}'>"
            f"<p class='caption'>{html.escape(name)}</p>"
            f"</div>"
        )

    rows += ["</body>", "</html>"]

    with open(output_html, 'w') as f:
        f.write('\n'.join(rows))

# Example usage
if __name__ == "__main__":
    
    # Generate random point clouds for demonstration
    import h5py
    # model_metas_1 = h5py.File(os.path.join('/mnt/lingjie_cache/sphere_traj/sphere/sphere_00001.h5'), 'r')
    # model_metas_2 = h5py.File(os.path.join('/mnt/lingjie_cache/sphere_traj/sphere/sphere_00002.h5'), 'r')
    model_metas_1 = h5py.File(os.path.join('/mnt/kostas-graid/datasets/chenwang/traj/sphere_traj_force0.1/sphere/sphere_13445.h5'), 'r')
    model_metas_2 = h5py.File(os.path.join('/mnt/kostas-graid/datasets/chenwang/traj/sphere_traj_force0.1/sphere/sphere_00002.h5'), 'r')
    model_pcls_1 = np.array(model_metas_1['x'])
    model_pcls_2 = np.array(model_metas_2['x'])

    num_drag_points = int(np.array(model_metas_1['drag_mask']).astype(np.float32).sum(axis=-1))
    import torch
    mask = torch.cat([torch.zeros(10000 - num_drag_points, dtype=torch.bool), torch.ones(num_drag_points, dtype=torch.bool)]).cpu().numpy()
    
    # Visualize in a 2x3 grid
    # vis_pcl_grid([model_pcls[0], model_pcls[30]], 'test.png', grid_shape=(1, 2))
    save_pointcloud_video(model_pcls_1, model_pcls_2, 'debug/visualize_pcls.gif', drag_mask=mask)