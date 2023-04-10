import cv2
import numpy as np
from sklearn.cluster import KMeans
import math
import colorsys

tester = []
for i in range(1, 8+1):
    tester.append(f'test{i}.png')

color_count = 64

def change_value(x):
        global cp
        # cp.get_path((255,255,255))

def mouse_click(event, x, y, flags, param):
    global cp
    if event == cv2.EVENT_FLAG_LBUTTON:
        pixel = cp.img[y, x]
        cp.pos_x = y
        cp.pos_y = x
        # cv2.setTrackbarPos("H1", "show", pixel[0])
        # cv2.setTrackbarPos("H2", "show", pixel[0])
        # cv2.setTrackbarPos("S1", "show", int(pixel[1]/10)*10)
        # cv2.setTrackbarPos("S2", "show", int(pixel[1]/10)*10+10)
        # cv2.setTrackbarPos("V1", "show", int(pixel[2]/10)*10)
        # cv2.setTrackbarPos("V2", "show", int(pixel[2]/10)*10+10)
        cp.get_path(cp.image[y, x])

class GroupInfo:
    def __init__(self, contours, color):
        self.contours = contours
        self.color = color

        self.area = 0
        for contour in contours:
            self.area += cv2.contourArea(contour)
    
    def get_path_tag(self, points, color):
        path_tag = '<path d="'
        path_tag = path_tag + f'M{points[0][0][0]} {points[0][0][1]} '
        for i in range(1, len(points), 2):
            if i > len(points):
                break
            pt = points[i][0]
            prev_pt = points[i-1][0]

            # x = (pt[0] + prev_pt[0])/2
            # y = (pt[1] + prev_pt[1])/2
            x = pt[0]
            y = pt[1]
            path_tag = path_tag + f'L{x} {y} '
        
        path_tag = path_tag + f'z" stroke = "rgb({color[0]},{color[1]},{color[2]})" fill="rgb({color[0]},{color[1]},{color[2]})"  stroke-width="1.5"/>'
        return path_tag
    
    def get_group_tag(self):
        color = self.color
        group_tag = f'<g>'
        for contour in self.contours:
            # refined_contour = self.refine(contour)
            # refined_contour = cv2.approxPolyDP(contour, cv2.arcLength(contour, True)*0.0001, True)
            path_tag = self.get_path_tag(contour, self.color)
            group_tag += path_tag
        group_tag += '</g>'

        return group_tag
    
    def refine(self, contour):
        ret_val = []
        if len(contour) <= 3:
            return contour
        for i in range(0, int(len(contour)/3), 3):
            pt1 = contour[i][0]
            pt2 = contour[i+1][0]
            # pt3 = contour[i+2][0]
            # print(len(contour), pt1, pt2, pt3)
            x1 = pt1[0]
            y1 = pt1[1]
            x2 = pt2[0]
            y2 = pt2[1]
            # x3 = pt3[0]
            # y3 = pt3[1]

            x = (x1+x2)/2
            y = (y1+y2)/2

            ret_val.append([[x1, y1]])
            # print('append', (x, y))
        return ret_val

class ColorPicker:
    
    def __init__(self, image_name):
        self.image = cv2.imread(image_name)
        # self.image = cv2.resize(self.image, [500,500])
        self.image, self.colors = self.kmeans_color_quantization(self.image, clusters=color_count)
        self.img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV) 
        
        self.title = image_name.split('.')[0]

        # self.img[:,:,0] = cv2.equalizeHist(self.img[:,:,0])
        self.pos_x = 0
        self.pos_y = 0
        self.accumulate_result = np.zeros(self.image.shape).astype(np.uint8)
        self.groups = []

    def kmeans_color_quantization(self, image, clusters=8, rounds=1):
        h, w = image.shape[:2]
        samples = np.zeros([h*w,3], dtype=np.float32)
        count = 0

        for x in range(h):
            for y in range(w):
                samples[count] = image[x][y]
                count += 1

        compactness, labels, centers = cv2.kmeans(samples,
                clusters, 
                None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
                rounds, 
                cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        return res.reshape((image.shape)), centers

    def rgb_to_hsv(self, r,g,b):
        rgb = np.array([[[b,g,r]]], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        hsv = np.squeeze(hsv)
        return hsv[0], hsv[1], hsv[2]

    def calc_dist(self, color1, color2):
         return math.sqrt( math.pow(color1[0] - color2[0], 2) + math.pow(color1[1] - color2[1], 2) + math.pow(color1[2] - color2[2], 2))

    def get_path(self, color):
        b, g, r = int(color[0]), int(color[1]), int(color[2])
        h, s, v = self.rgb_to_hsv(r,g,b)

        h1 = h
        h2 = h
        s1 = int(s)
        s2 = int(s)
        v1 = int(v)
        v2 = int(v)
        # hsvLower = np.array([h1, s1, v1])
        # hsvUpper = np.array([h2, s2, v2])

        

        # h1 = cv2.getTrackbarPos("H1", "show")
        # s1 = cv2.getTrackbarPos("S1", "show")
        # v1 = cv2.getTrackbarPos("V1", "show")

        # h2 = cv2.getTrackbarPos("H2", "show")
        # s2 = cv2.getTrackbarPos("S2", "show")
        # v2 = cv2.getTrackbarPos("V2", "show")

        print(f'rgb : {r}, {g}, {b}')
        print(f'calc_hsv : {h}, {s}, {v}')
        print(f'real_hsv : {h1}, {s1}, {v1}')
        print()
        # print(h1, s1, v1, h2, s2, v2)
        hsvLower = np.array([h1, s1, v1])
        hsvUpper = np.array([h2, s2, v2])

        # hsvLower = np.array([h, s, v])
        # hsvUpper = np.array([h, s, v])

        hsv_mask = cv2.inRange(self.img, hsvLower, hsvUpper)

        kernel3 = np.ones((3, 3), np.uint8)
        bit_and_result = cv2.bitwise_and(self.image, self.image, mask=hsv_mask)
        
        bit_and_close = cv2.morphologyEx(bit_and_result, cv2.MORPH_CLOSE, kernel3)
        # bit_and_open = cv2.morphologyEx(bit_and_close, cv2.MORPH_OPEN, kernel3)

        bit_and_result_gray = cv2.cvtColor(bit_and_close, cv2.COLOR_BGR2GRAY)
        # bit_and_result_blur = cv2.GaussianBlur(bit_and_result_gray, (0, 0), 0.01)

        contours, hier = cv2.findContours(bit_and_result_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        self.accumulate_result = cv2.drawContours(self.accumulate_result, contours, -1, [b,g,r], -1)

        self.groups.append(GroupInfo(contours, [r,g,b] ))
        # for contour in contours:
        #     path_tag = self.generate_path(contour, color)
        #     self.path_tags.append(path_tag)
        
        # cv2.imshow("bitwise_and", bit_and_result)
        # cv2.imwrite(f"({r},{g},{b})_({h},{s},{v}).png", bit_and_result)
    
    def get_svg(self):
        for color in self.colors:
            self.get_path(color)
        cv2.imwrite("acc.png", self.accumulate_result)
        
        print("Generate SVG")

        height = self.image.shape[0]
        width = self.image.shape[1] 
        svg_file = f'<svg height="{height}" width="{width}" id="outputsvg"  xmlns="http://www.w3.org/2000/svg"  style="transform: none;   cursor: move;" viewBox="0 0 {height} {width}">'
        
        self.groups = sorted(self.groups, key=lambda group: group.area, reverse=True)

        for group in self.groups:
            svg_file += group.get_group_tag()

        svg_file += f'</svg>'

        with open(f"{self.title}_output.svg", "w") as file:
            file.write(svg_file)
        
    

        



cp = []
for test in tester:
    cp.append(ColorPicker(test))

if __name__ == "__main__":
    for c in cp:
        c.get_svg()
    