import gradio as gr
from PIL import Image
import numpy as np

from copy import deepcopy
import cv2
import plotly.graph_objects as go



def mask_image(image,
               mask,
               color=[255,0,0],
               alpha=0.5):
    """ Overlay mask on image for visualization purpose.
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    return out

def image_preprocess(input_image, target_res, lower_contrast=True, rescale=True):
    image_arr = np.array(input_image)
    in_w, in_h = image_arr.shape[:2]

    if lower_contrast:
        alpha = 0.8  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)
        # Apply the contrast adjustment
        image_arr = cv2.convertScaleAbs(image_arr, alpha=alpha, beta=beta)
        image_arr[image_arr[..., -1] > 200, -1] = 255

    ret, mask = cv2.threshold(
        np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY
    )
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    ratio = 0.75
    if rescale:
        side_len = int(max_size / ratio)
    else:
        side_len = in_w
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2
    padded_image[
        center - h // 2 : center - h // 2 + h, center - w // 2 : center - w // 2 + w
    ] = image_arr[y : y + h, x : x + w]
    rgba = Image.fromarray(padded_image).resize((target_res, target_res), Image.LANCZOS)
    return y + h // 2 - center, x + w // 2 - center, side_len, rgba

def plot_point_cloud(points, arrows):
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue'),
        name='Point Cloud'
    )

    cone_traces = []
    for arrow in arrows:
        origin = arrow['origin']
        direction = arrow['dir']
        cone = go.Cone(
            x=[origin[0]], y=[origin[1]], z=[origin[2]],
            u=[direction[0]], v=[direction[1]], w=[direction[2]],
            sizemode='raw', sizeref=5,
            anchor='tail', colorscale='Reds', showscale=False,
            name='Arrow'
        )
        cone_traces.append(cone)

    fig = go.Figure(data=[scatter] + cone_traces)
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=0, b=0), height=400)
    return fig
