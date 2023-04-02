import cv2
import numpy as np
image = cv2.imread("palette.png")
image = cv2.resize(image, [500, 500])
img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
img[:,:,0] = cv2.equalizeHist(img[:,:,0])
pos_x = 0
pos_y = 0


accumulate_result = np.zeros(image.shape).astype(np.uint8)

def change_value(x):
    global accumulate_result
    _, accumulate_result = get_path(accumulate_result)

def mouse_click(event, x, y, flags, param):
    global accumulate_result
    global pos_x
    global pos_y
    if event == cv2.EVENT_FLAG_LBUTTON:
        pixel = img[y, x]
        cv2.setTrackbarPos("H1", "show", pixel[0])
        cv2.setTrackbarPos("H2", "show", pixel[0])
        cv2.setTrackbarPos("S1", "show", int(pixel[1]/10)*10)
        cv2.setTrackbarPos("S2", "show", int(pixel[1]/10)*10+10)
        cv2.setTrackbarPos("V1", "show", int(pixel[2]/10)*10)
        cv2.setTrackbarPos("V2", "show", int(pixel[2]/10)*10+10)
        pos_x = y
        pos_y = x
        _, accumulate_result = get_path(accumulate_result)
        

def get_path(accumulate_result):
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

    contour_result = np.zeros((image.shape[0], image.shape[1], 3))
    color = image[pos_x, pos_y]
    b, g, r = int(color[0]), int(color[1]), int(color[2])
    print(pos_x, pos_y, b, g,r)

    cv2.drawContours(contour_result, contours, -1, (b,g,r))

    
    try:
        accumulate_result = cv2.drawContours(accumulate_result, contours, -1, [b,g,r], -1)
    except:
        accumulate_result = np.zeros((image.shape[0], image.shape[1], 3))
        accumulate_result = cv2.drawContours(accumulate_result, contours, -1, [b,g,r], -1)
    cv2.imshow("bitwise_and", bit_and_result)
    cv2.imshow("mask", hsv_mask)
    cv2.imshow("contour", contour_result)
    cv2.imshow("acc", accumulate_result)

    return contours, accumulate_result

if __name__ == "__main__":
    
    # cv2.cvtColor(accumulate_result, cv2.COLOR_GRAY2BGR)

    cv2.namedWindow("show")
    cv2.namedWindow("bitwise_and")
    cv2.namedWindow("mask")
    cv2.namedWindow("acc")
    cv2.namedWindow("contour")
    cv2.createTrackbar("H1", "show", 0, 255, change_value)
    cv2.createTrackbar("S1", "show", 0, 255, change_value)
    cv2.createTrackbar("V1", "show", 0, 255, change_value)

    cv2.createTrackbar("H2", "show", 0, 255, change_value)
    cv2.createTrackbar("S2", "show", 0, 255, change_value)
    cv2.createTrackbar("V2", "show", 0, 255, change_value)

    cv2.setMouseCallback("show", mouse_click)
    
    cv2.imshow("show", image)

    while True:
        cv2.waitKey(1)

    '''
    @Todo
    대표 색을 추출하든, 5px 단위로 이동하면서 확인하든
    이미지 내의 각 색상에 대한 처리 결과를 하나로 합쳐서
    최종 svg 파일을 생성
    '''

    cv2.destroyAllWindows()