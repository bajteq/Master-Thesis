from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import os
from PIL import Image


path = 'tumor-segmentation\\static\\'

image_file_path = 'tumor-segmentation\\static\\49.png'
generated_mask = 'tumor-segmentation\\static\\generated_mask1.png'


def blend_images(brain, mask):
    background = Image.open(brain)
    overlay = Image.open(mask)

#     background = background.convert("RGBA")
#     overlay = overlay.convert("RGBA")

    background = np.asarray(background)
    overlay = np.asarray(overlay)
    overlay_copy = overlay.copy()
    overlay_copy.setflags(write=1)
    for x in range(128):
        for y in range(128):
            if not np.array_equal(overlay_copy[x, y], np.array([0, 0, 0, 255])):
                overlay_copy[x, y] = np.array([255, 0, 0, 255])

    background = Image.fromarray(background)
    overlay = Image.fromarray(overlay_copy)
    new_img = Image.blend(background, overlay, 0.5)
    blended_img_path = "tumor-segmentation\\static\\blend.png"
    new_img.save(blended_img_path, "PNG")


blend_images(image_file_path, generated_mask)
