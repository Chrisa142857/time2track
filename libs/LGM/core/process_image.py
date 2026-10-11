import os 
import rembg
import numpy as np
import cv2

from PIL import Image

def recenter(image, h_begin=100, w_begin=220, res=256):
    h_image, w_image = image.shape[:2]
    new_image = np.zeros((res, res, 4), dtype=np.uint8) 
    h_begin_new = -min(0, h_begin)
    w_begin_new = -min(0, w_begin)
    if h_begin > 0 and w_begin > 0:
        new_image = image[h_begin:h_begin+res, w_begin:w_begin+res]
    else:
        new_image[h_begin_new:h_begin_new+h_image, w_begin_new:w_image] = image
    return new_image

def recover(image, original_size=(720, 480), h_begin=100, w_begin=220, res=256):
    target_w, target_h = original_size
    recovered_image = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    h_begin_new = -min(0, h_begin)
    w_begin_new = -min(0, w_begin)
    if h_begin > 0 and w_begin > 0: 
        recovered_image[h_begin:h_begin+res, w_begin:w_begin+res] = image
    else:
        recovered_image = image[h_begin_new:h_begin_new+target_h, w_begin_new:w_begin_new+target_w]
    return recovered_image.astype(np.uint8)

def resize_and_center_crop(image, target_h=480, target_w=720):
    
    w, h = image.size 
    image_ratio = w / h
    
    if target_w / target_h > image_ratio:
        new_w = target_w
        new_h = int(h * (target_w / w))
    else:
        new_h = target_h
        new_w = int(w * (target_h / h))
        
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left = max(0, (new_w - target_w) // 2)
    top = max(0, (new_h - target_h) // 2)
    right = left + target_w
    bottom = top + target_h
    image = image.crop((left, top, right, bottom))
    return image

if __name__ == "__main__":
    
    base_dir = 'data_test'
    task_name = 'plane'
    raw_path = os.listdir(f'{base_dir}/raw_data')
    bg_remover = rembg.new_session()
    
    for image_path in raw_path: 
        if not f'{task_name}_original' in image_path:
            continue
        input_image = Image.open(f'{base_dir}/raw_data/{image_path}')
        image = resize_and_center_crop(input_image)
        image.save(f'{base_dir}/raw_data/{image_path[:-4]}_resized.png')
        image.save(f'{base_dir}/{image_path.split("_")[0]}.png')
        image = np.array(image)
        
        carved_image = rembg.remove(image, session=bg_remover) # [H, W, 4]
        Image.fromarray(carved_image).save(f'{base_dir}/raw_data/{image_path[:-4]}_carved.png')
        
        ### Test
        # mask = carved_image[..., -1] > 0
        # image = recenter(carved_image)
        # image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        # Image.fromarray(image).save(f'{base_dir}/raw_data/{image_path[:-4]}_recentered.png')
        # image = cv2.resize(image, (280, 280), interpolation=cv2.INTER_AREA)
        # image = recover(image, (720, 480))
        # Image.fromarray(image).save(f'{base_dir}/raw_data/{image_path[:-4]}_recovered.png')
        