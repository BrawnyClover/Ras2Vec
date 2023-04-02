import cv2
import numpy as np
image = cv2.imread("test4.png")
img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
img[:,:,0] = cv2.equalizeHist(img[:,:,0])

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        pixel = img[y, x]
        cv2.setTrackbarPos("H1", "show", pixel[0])
        cv2.setTrackbarPos("H2", "show", pixel[0]+10)
        cv2.setTrackbarPos("S1", "show", 10)
        cv2.setTrackbarPos("S2", "show", 245)
        cv2.setTrackbarPos("V1", "show", 10)
        cv2.setTrackbarPos("V2", "show", 255)

if __name__ == "__main__":
    

    cv2.namedWindow("show")
    cv2.createTrackbar("H1", "show", 0, 255, lambda x:x)
    cv2.createTrackbar("S1", "show", 0, 255, lambda x:x)
    cv2.createTrackbar("V1", "show", 0, 255, lambda x:x)

    cv2.createTrackbar("H2", "show", 0, 255, lambda x:x)
    cv2.createTrackbar("S2", "show", 0, 255, lambda x:x)
    cv2.createTrackbar("V2", "show", 0, 255, lambda x:x)

    cv2.setMouseCallback("show", mouse_click)
    

    while cv2.waitKey(1) != ord('q'):
        h1 = cv2.getTrackbarPos("H1", "show")
        s1 = cv2.getTrackbarPos("S1", "show")
        v1 = cv2.getTrackbarPos("V1", "show")

        h2 = cv2.getTrackbarPos("H2", "show")
        s2 = cv2.getTrackbarPos("S2", "show")
        v2 = cv2.getTrackbarPos("V2", "show")
        # print(h1, s1, v1, h2, s2, v2)
        hsvLower = np.array([h1, s1, v1])
        hsvUpper = np.array([h2, s2, v2])

        hsv_mask = cv2.inRange(img, hsvLower, hsvUpper)

        kernel3 = np.ones((3, 3), np.uint8)
        # kernel5 = np.ones((5, 5), np.uint8)
        
        # hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel3)
        hsv_mask_dil = cv2.dilate(hsv_mask, kernel3)
        hsv_mask_erode = cv2.erode(hsv_mask, kernel3)
        hsv_mask = cv2.bitwise_xor(hsv_mask_erode, hsv_mask_dil)
        # hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel3)
        # hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel3)
        # hsv_mask_dil = cv2.dilate(hsv_mask, kernel3)
        # hsv_mask = cv2.bitwise_xor(hsv_mask_dil, hsv_mask)
        bit_and_result = cv2.bitwise_and(image, image, mask=hsv_mask)
        bit_and_result_gray = cv2.cvtColor(bit_and_result, cv2.COLOR_BGR2GRAY)
        # bit_and_result = cv2.morphologyEx(bit_and_result, cv2.MORPH_CLOSE, kernel3)

        contours, hier = cv2.findContours(bit_and_result_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_result = np.zeros(image.shape)
        cv2.drawContours(contour_result, contours, -1, (255,0,255))
        cv2.imshow("mask", hsv_mask)
        cv2.imshow("and", bit_and_result)
        cv2.imshow("show", contour_result)

    cv2.destroyAllWindows()