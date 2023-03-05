import cv2
import ras2vec


if __name__ == "__main__":
    image = cv2.imread('test3.png')
    toVecConverter = ras2vec.Ras2vec_Converter(image)
    # path_list = toVecConverter.convert()
    ret = toVecConverter.convert()
    cv2.imwrite('ret3.png', ret)