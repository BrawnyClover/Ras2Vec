import numpy as np
from collections import deque
import math
import random

class Ras2vec_Converter:
    def __init__(self, image):
        self.image = image
        self.height, self.width, _= self.image.shape
        self.check = np.zeros((self.height, self.width), dtype=np.uint8)
        self.boundary = np.zeros((self.height, self.width), dtype=np.uint8)
        self.boundary_point = list()
        self.count = 1
        self.pixel_cnt = self.height * self.width
    
    def convert(self):
        for i in range(0, self.height):
            for j in range(0, self.width):
                if self.check[i, j] == 0:
                    self.bfs((i, j), self.image[i, j])
                    self.count += 1
        
        ret = self.point_to_path()
        # ret = self.check2img()
        return ret
    
    def generate_path(self, points, color):
        path_tag = '<path d="'
        path_tag = path_tag + f'M{points[0][1]} {points[0][0]} '
        prev_point = None
        for i in range(1, len(points), 1):
            if i > len(points):
                break
            # if prev_point is None:
            #     prev_point = points[i]
            path_tag = path_tag + f'L{points[i][1]} {points[i][0]} '
        
        path_tag = path_tag + f'z" stroke = "rgb({color[2]},{color[1]},{color[0]})"/>'
        return path_tag
            
    def sort_point(self, points):
        ret_val = []
        ret_val.append(points[0])
        last_point = points[0]
        check = [0]*len(points)
        check_cnt = 0
        for i in range(0, len(points)):
            min_dist = 999
            min_point = 0
            for i in range(0, len(points)):
                if check[i] != 1:
                    dist = self.calc_real_dist(last_point, points[i])
                    if dist < min_dist:
                        min_dist = dist
                        min_point = i
            check[min_point] = 1
            ret_val.append(points[min_point])
            last_point = points[min_point]
        
        return ret_val


    def point_to_path(self):
        svg_file = f'<svg height="{self.height}" width="{self.width}" id="outputsvg"  xmlns="http://www.w3.org/2000/svg"  style="transform: none;   cursor: move;" viewBox="0 0 {self.height} {self.width}">  <g fill="none" stroke="black" stroke-width="2">'
        for points, color in self.boundary_point:
           sorted = self.sort_point(points)
           svg_file += self.generate_path(sorted, color)
        svg_file += f'</g></svg>'

        with open("output.svg", "w") as file:
            file.write(svg_file)

    def calc_real_dist(self, pt1, pt2):
            # print(f'{pixel1}, {pixel2}')
            dist = math.sqrt( math.pow((int(pt1[0])-int(pt2[0])), 2) + math.pow(int(pt1[1])-int(pt2[1]), 2))
            return dist

    def calc_dist(self, pixel1, pixel2):
        # print(f'{pixel1}, {pixel2}')
        dist = math.sqrt( math.pow((int(pixel1[0])-int(pixel2[0])), 2) + math.pow(int(pixel1[1])-int(pixel2[1]), 2) + math.pow(int(pixel1[2])-int(pixel2[2]), 2))
        return dist

    def bfs(self, coord, pixel):
        queue = deque([coord])
        self.check[coord[0], coord[1]] = self.count
        # print(f'{coord}, {self.count}, {self.pixel_cnt}')
        threshold = 100
        points = list()

        while queue:
            print(f'count : {self.count} / {self.pixel_cnt}')
            c_coord = queue.popleft()
            # print('pop : {}, last count : {}, current count : {}'.format(c_coord, len(queue), self.check[c_coord]))
            if self.check[c_coord] == 0:
                break
            weight = [(-1, 0), (0, -1), (0, 1), (1, 0)]
            for i, j in weight:
                next_coord_i = c_coord[0] + i
                next_coord_j = c_coord[1] + j
                if next_coord_i < 0 or next_coord_i == self.height:
                    continue
                if next_coord_j < 0 or next_coord_j == self.width:
                    continue
                next_pixel = self.image[next_coord_i, next_coord_j]

                # print('{}, {}, {}, {}'.format(c_coord, (next_coord_i, next_coord_j), self.check[next_coord_i, next_coord_j], self.count))

                if self.check[next_coord_i, next_coord_j] == 0:
                    if self.calc_dist(pixel, next_pixel) < threshold:
                        self.check[next_coord_i, next_coord_j] = self.count
                        queue.append((next_coord_i, next_coord_j))
                        # print('from {}, push {}, count {}'.format(c_coord, (next_coord_i, next_coord_j), self.count))

                    else:
                        self.boundary[c_coord] = self.count
                        points.append(c_coord)    
        # print("add")
        if(len(points) > 0):
            self.boundary_point.append([points, self.image[points[0]]])
            

    def check2img(self):
        colors = []
        ret = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for i in range(self.count):
            colors.append((random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256)))
        colors[0] = (255, 255, 255)
        for i in range(self.height):
            for j in range(self.width):
                ret[i, j] = colors[self.boundary[i, j]]
        
        return ret