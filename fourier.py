import numpy as np
import cv2

def get_fft(image):

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    rows,cols = image.shape
    c_row, c_col = round(rows/2), round(cols / 2)

    mask_width = round((rows / 5) / 2)
    mask_height = round((cols / 5) / 2)

    fshift[c_row-mask_width:c_row+mask_width, c_col-mask_height:c_col+mask_height] = 0

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    result = np.abs(img_back)

    return result

def convert(image):
    blue, green, red = cv2.split(image)
    f_b = get_fft(blue)
    f_g = get_fft(green)
    f_r = get_fft(red)
    cv2.imwrite("r.png", f_r)
    cv2.imwrite("g.png", f_g)
    cv2.imwrite("b.png", f_b)

    rgb = cv2.merge((f_b, f_g, f_r))
    return rgb