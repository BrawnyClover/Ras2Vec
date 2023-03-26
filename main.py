import cv2
import ras2vec
import numpy as np
from matplotlib import pyplot as plt
import fourier


if __name__ == "__main__":
    file_idx = 3
    file_name = f'test{file_idx}.png'
    fft_res_name = f'fft_ret{file_idx}.png'
    bfs_res_name = f'bfs_ret{file_idx}.png'

    image = cv2.imread(file_name)
    fft_res = fourier.convert(image)
    bfs_res = ras2vec.Ras2vec_Converter(image).convert()

    

    
    # # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # # rgb = rgb / np.float32(255)
    
    # img_original = cv2.hconcat([blue, green])
    # img_original = cv2.hconcat([img_original, red])

    # img_res = cv2.hconcat([f_b, f_g])
    # img_res = cv2.hconcat([img_res, f_r])
    
    cv2.imwrite(fft_res_name, fft_res)
    # cv2.imwrite(bfs_res_name, bfs_res)
    # cv2.imshow("3",img_original)
    # cv2.imshow("4", img_res)
    
    # plt.subplot(121),plt.imshow(img_original, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(rgb, cmap=)
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.show()