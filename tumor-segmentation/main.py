from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess(image_path):
    siz = 128
    img = plt.imread(image_path)
    if len(img.shape) > 2:
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    img = np.asarray(img)
    img = resize(img, (siz, siz), preserve_range=True)
    img = img.astype('float32')
    plt.imsave(image_path, img, cmap='gray')
    return img


def prep_matrix(image):
    siz = 128
    zeros_arr_mask = np.zeros((1, siz, siz), dtype=np.float32)
    zeros_arr_mask[0, :, :] = image
    Y = np.expand_dims(zeros_arr_mask, axis=3)
    return Y


def prediction(image, model, model_no, root_path):
    pred = model.predict(image, verbose=1)
    pred = pred[:, :, :, 0]
    pred = pred > 0.5
    mask = pred[0, :, ]
    path_for_mask = root_path+'/static/' + \
        "generated_mask"+model_no+".png"
    plt.imsave(path_for_mask, mask, cmap='gray')
    return path_for_mask


def blend_images(brain_path, mask_path, model_no, root_path):
    background = Image.open(brain_path)
    overlay = Image.open(mask_path)

    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

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

    blended_img_path = root_path+'/static/' + \
        "blended_image"+model_no+".png"
    new_img.save(blended_img_path, "PNG")
    return blended_img_path


def delete_files(files):
    if len(files) !=0:
        for f in files:
            os.remove(f)