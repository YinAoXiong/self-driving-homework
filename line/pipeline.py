# -*- coding: utf-8 -*-

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from . import line
from . import utils


def thresholding(img):
    # setting all sorts of thresholds
    x_thresh = utils.abs_sobel_thresh(
        img, orient='x', thresh_min=10, thresh_max=230)
    mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 150))
    dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    hls_thresh = utils.hls_select(img, thresh=(180, 255))
    lab_thresh = utils.lab_select(img, thresh=(155, 200))
    luv_thresh = utils.luv_select(img, thresh=(225, 255))

    # Thresholding combination
    threshholded = np.zeros_like(x_thresh)
    threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (
        hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1

    return threshholded


def processing(img, object_points, img_points, M, Minv, left_line, right_line):
    # camera calibration, image distortion correction
    undist = utils.cal_undistort(img, object_points, img_points)
    # get the thresholded binary image
    thresholded = thresholding(undist)

    # perform perspective  transform
    thresholded_wraped = cv2.warpPerspective(
        thresholded, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    # plt.clf()
    # plt.imshow(thresholded_wraped, cmap ='gray')
    # plt.draw()
    # plt.pause(3)

    print("thresholded的shape", thresholded_wraped.shape)
    # perform detection
    if left_line.detected and right_line.detected:
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line_by_previous(
            thresholded_wraped, left_line.current_fit, right_line.current_fit)
    else:
        left_fit, right_fit, left_lane_inds, right_lane_inds, left_line.line_type, right_line.line_type = utils.find_line(
            thresholded_wraped)
    left_line.line_color, right_line.line_color = utils.find_line_color(
        undist, thresholded_wraped, M, img, left_line.line_type, right_line.line_type)
    left_line.update(left_fit)
    right_line.update(right_fit)

    # draw the detected laneline and the information
    area_img = utils.draw_area(
        undist, thresholded_wraped, Minv, left_fit, right_fit)
    curvature, pos_from_center = utils.calculate_curv_and_pos(
        thresholded_wraped, left_fit, right_fit)
    result = utils.draw_values(area_img, curvature, pos_from_center, left_line.line_type,
                               right_line.line_type, left_line.line_color, right_line.line_color)

    return result


class LINE(object):
    def __init__(self):
        self.left_line = line.Line()
        self.right_line = line.Line()
        self.cal_imgs = utils.get_images_by_dir(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'camera_cal'))
        self.object_points, self.img_points = utils.calibrate(
            self.cal_imgs, grid=(9, 6))
        self.M, self.Minv = utils.get_M_Minv()

        
    def calculate_img(self, img):
        result = processing(img, self.object_points, self.img_points,
                            self.M, self.Minv, self.left_line, self.right_line)
        return result
