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
        ret = self.check2img()
        return ret
    
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
            c_coord = queue.popleft()
            print('pop : {}, last count : {}, current count : {}'.format(c_coord, len(queue), self.check[c_coord]))
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
                        print('from {}, push {}, count {}'.format(c_coord, (next_coord_i, next_coord_j), self.count))

                    else:
                        self.boundary[c_coord] = self.count
                        points.append(c_coord)    
        
        self.boundary_point.append(points)
            

    def check2img(self):
        colors = []
        ret = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for i in range(self.count):
            colors.append((random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256)))
        colors[0] = (255, 255, 255)
        for i in range(self.height):
            for j in range(self.width):
                ret[i, j] = colors[self.check[i, j]]
        
        return ret