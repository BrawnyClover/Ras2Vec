import cv2
import numpy as np

image = cv2.imread("77.png")
print(image)
kernel3 = np.ones((3, 3), np.uint8)
# kernel3 = np.ones((3, 3), np.uint8)
# kernel3[0, 0] = 0
# kernel3[2, 0] = 0
# kernel3[0, 2] = 0
# kernel3[2, 2] = 0
dil = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel3)
ero = cv2.morphologyEx(dil, cv2.MORPH_ERODE, kernel3)
result = cv2.hconcat([dil, ero])
# result = cv2.dilate(image, kernel3)
cv2.imwrite("77res.png", result)