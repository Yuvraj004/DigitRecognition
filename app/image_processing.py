import numpy as np
from skimage import exposure # type: ignore
from PIL import Image, ImageOps, ImageChops
from io import BytesIO
import base64

def replace_transparent_background(image):
    
    image_arr = np.array(image)
    print(f"Shape of image_arr: {image_arr.shape}")
    print(f"Image dimensions: {image_arr.ndim}")

    if len(image_arr.shape) <= 2:
        return image_arr

    alpha1 = 0
    r2, g2, b2, alpha2 = 255, 255, 255, 255

    red, green, blue, alpha = image_arr[:, :, 0], image_arr[:, :, 1], image_arr[:, :, 2], image_arr[:, :, 3]
    mask = (alpha == alpha1)
    image_arr[:, :, :4][mask] = [r2, g2, b2, alpha2]

    return Image.fromarray(image_arr)

def trim_borders(image):
    bg = Image.new("P", image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)
    
    return image

def pad_image(image):
    return ImageOps.expand(image, border=30, fill='#fff')


def to_grayscale(image):
    return image.convert('L')

def invert_colors(image):
    return ImageOps.invert(image)

def resize_image(image):
    return image.resize((8, 8), Image.LINEAR) # type: ignore

def process_image(image):
    img  = replace_transparent_background(image)
    img = trim_borders(img)
    img = pad_image(img)
    img = to_grayscale(img)
    img_inv =invert_colors(img)
    img_final = resize_image(img_inv)
    
    return Image.fromarray(img_final)
