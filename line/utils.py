# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#√ get all image in the given directory persume that this directory only contain image files
def get_images_by_dir(dirname):
    img_names = os.listdir(dirname)
    img_paths = [dirname+'/'+img_name for img_name in img_names]

    imgs = [cv2.imread(path) for path in img_paths]
    return imgs

#function take the chess board image and return the object points and image points
def calibrate(images,grid=(9,6)):
    object_points=[]
    img_points = []
    count=0
    for img in images:
        count=count+1
        object_point = np.zeros( (grid[0]*grid[1],3),np.float32 )
        object_point[:,:2]= np.mgrid[0:grid[0],0:grid[1]].T.reshape(-1,2)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, grid, None)
        if ret:
            object_points.append(object_point)
            img_points.append(corners)
    return object_points,img_points

def get_M_Minv():
    src = np.float32([[(203, 720), (585, 460), (695, 460), (1127, 720)]])
    dst = np.float32([[(320, 720), (320, 0), (960, 0), (960, 720)]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    return M,Minv
    
#function takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

# 颜色梯度阈值过滤
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    print("abs_sobel_thresh")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(gray.shape)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 算子sobelx sobely得出来的即是阈值过滤后的边界
    # Calculate the gradient magnitude
    # 和上面的abs_sobel_thresh只差在这一点
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_select(img,channel='s',thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel=='h':
        channel = hls[:,:,0]
    elif channel=='l':
        channel=hls[:,:,1]
    else:
        channel=hls[:,:,2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output

def luv_select(img, thresh=(0, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output

def lab_select(img, thresh=(0, 255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b_channel = lab[:,:,2]
    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1
    return binary_output

# √
def find_line(binary_warped):
    print("find_line")
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # plt.figure("lena")
    # n, bins, patches = plt.hist(histogram)
    # plt.show()
    # plt.draw()
    # plt.pause(3)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # 得到左右两边峰值部分
    # Choose the number of sliding windows
    nwindows = 9    #大概是滑动九次的意思
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    print("shi binary_warped ")
    print(binary_warped.shape)
    print("nonzero  :",nonzero)
    print(binary_warped[0][358],binary_warped[358][0])
    nonzeroy = np.array(nonzero[0])   #y指的第几行
    nonzerox = np.array(nonzero[1])
    print("nonzero y x shape  :",nonzeroy.shape,nonzerox.shape)
    print(nonzeroy,nonzerox)            #这里好象搞反了
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        if window==2:
            print("这是什么鬼操作")
            print(good_left_inds.shape,good_left_inds)
            print("窗口大小范围",win_y_low,win_y_high,win_xleft_low,win_xleft_high)
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    num_lefty = np.unique(lefty).shape[0]/binary_warped.shape[0]
    num_righty = np.unique(righty).shape[0]/binary_warped.shape[0]
    percent = 2/3
    print("窗口大小以及左右两边车道线覆盖的y值个数：",binary_warped.shape[0],num_lefty,num_righty)
    if num_lefty > percent:
        left_line_type="Solid"
    else:
        left_line_type="Dotted"
    if num_righty > percent:
        right_line_type="Solid"
    else:
        right_line_type="Dotted"

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)    #参数相当于按顺序（x,y,次幂）
    right_fit = np.polyfit(righty, rightx, 2)
    # print("二次多项式拟合：",left_fit)
    return left_fit, right_fit, left_lane_inds, right_lane_inds,left_line_type,right_line_type
# √
def find_line_by_previous(binary_warped,left_fit,right_fit):
    print("find_line_by_previous")
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    predict_left = fit_poly(left_fit,nonzeroy)
    predict_right = fit_poly(right_fit,nonzeroy)
    left_lane_inds = ((nonzerox > (fit_poly(left_fit,nonzeroy) - margin)) & (nonzerox < (fit_poly(left_fit,nonzeroy) + margin)))

    right_lane_inds = ((nonzerox > (fit_poly(right_fit,nonzeroy) - margin)) & (nonzerox < (fit_poly(right_fit,nonzeroy) + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit, left_lane_inds, right_lane_inds


# 添加车道线颜色
def find_line_color(undist,binary_warped,M,img,left_line_type,right_line_type):
    # histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # midpoint = np.int(histogram.shape[0]/2)

    color_wraped = cv2.warpPerspective(undist, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    yellow_left_mask = select_yellow(color_wraped,"left")
    yellow_right_mask = select_yellow(color_wraped,"right")
    white_left_mask = select_white(color_wraped,"left")
    white_right_mask = select_white(color_wraped,"right")
    # cv2.imshow("right0",yellow_left_mask)
    # cv2.waitKey()
    # cv2.imshow("right0",white_left_mask)
    # cv2.waitKey()
    # cv2.imshow("right0",yellow_right_mask)
    # cv2.waitKey()
    # cv2.imshow("right0",white_right_mask)
    # cv2.waitKey()

    yellow_left_num = np.count_nonzero(yellow_left_mask)
    yellow_right_num = np.count_nonzero(yellow_right_mask)
    white_left_num = np.count_nonzero(white_left_mask)
    white_right_num = np.count_nonzero(white_right_mask)
    print("left ::::::::::::::::::::::::",yellow_left_num,white_left_num)
    print("right    ::::::::::::::::::::",yellow_right_num,white_right_num)
    if yellow_left_num>white_left_num:
        left_line_color = "yellow"
    else:
        left_line_color = "white"
    if yellow_right_num>white_right_num:
        right_line_color = "yellow"
    else:
        right_line_color = "white"

    # yellow_nonzero = yellow_mask.nonzero()[1]
    # yellow_average = np.sum(yellow_nonzero)/yellow_nonzero.shape[0]
    # yellow_num = np.count_nonzero(yellow_mask)
    # print("yellow_average ",yellow_average,yellow_num,midpoint)
    return left_line_color,right_line_color


def draw_area(undist, binary_warped, Minv, left_fit, right_fit):
    print("draw_area")
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx_min = fit_poly(left_fit,ploty)-2
    left_fitx_max = fit_poly(left_fit,ploty)+10
    right_fitx_min = fit_poly(right_fit,ploty)-2
    right_fitx_max = fit_poly(right_fit,ploty)+10

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    print("color   shape   :::", warp_zero.shape, color_warp.shape, color_warp[0, 0].shape)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left_min = np.array([np.transpose(np.vstack([left_fitx_min, ploty]))])  # 本来是【0】为x,【1】存y 转置一下
    pts_left_max = np.array([np.flipud(np.transpose(np.vstack([left_fitx_max, ploty])))])
    pts_left = np.hstack((pts_left_min, pts_left_max))

    pts_right_min = np.array([np.transpose(np.vstack([right_fitx_min, ploty]))])  # 本来是【0】为x,【1】存y 转置一下
    pts_right_max = np.array([np.flipud(np.transpose(np.vstack([right_fitx_max, ploty])))])
    pts_right = np.hstack((pts_right_min, pts_right_max))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts_left]), (255, 38, 254))
    cv2.fillPoly(color_warp, np.int_([pts_right]), (255, 38, 254))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 1.5, 0)
    return result

# √
def calculate_curv_and_pos(binary_warped,left_fit, right_fit):
    print("calculate_curv_and_pos")
    # Define y-value where we want radius of curvature   也就是说把两条线上面所有的点都找出来了
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )  #整个y轴都需要
    leftx = fit_poly(left_fit,ploty)
    rightx = fit_poly(right_fit,ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature             √
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    curvature = ((left_curverad + right_curverad) / 2)
    #print(curvature)
    lane_width = np.absolute(leftx[719] - rightx[719])
    lane_xm_per_pix = 3.7 / lane_width
    veh_pos = (((leftx[719] + rightx[719]) * lane_xm_per_pix) / 2.)
    cen_pos = ((binary_warped.shape[1] * lane_xm_per_pix) / 2.)
    distance_from_center = cen_pos - veh_pos            #车道线中心和真实图像中心之间的偏移距离
    return curvature,distance_from_center

def select_yellow(image,side):
    print("select_yellow")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if side == "left":
        hsv = hsv[:,0:hsv.shape[1]//2]
    else:
        hsv = hsv[:,hsv.shape[1]//2:]
    # lower = np.array([11,43,46])
    # upper = np.array([25,255, 240])
    lower = np.array([11,54,46])
    upper = np.array([25,255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def select_white(image,side):
    print("select_white")
    if side == "left":
        image = image[:,0:image.shape[1]//2]
    else:
        image = image[:,image.shape[1]//2:]
    # hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([170,170,170])
    upper = np.array([255,255,255])
    mask = cv2.inRange(image, lower, upper)
    
    return mask

def fit_poly(para,x):
    z = np.poly1d(para)
    y = z(x)
    return y

def draw_values(img,curvature,distance_from_center,left_line_type,right_line_type,left_line_color,right_line_color):
    print("draw_values")
    font = cv2.FONT_HERSHEY_SIMPLEX
    radius_text = "Radius of Curvature: %sm"%(round(curvature))
    if distance_from_center>0:
        pos_flag = 'right'
    else:
        pos_flag= 'left'
    cv2.putText(img,radius_text,(100,100), font, 1,(255,255,255),2)
    center_text = "Vehicle is %.3fm %s of center"%(abs(distance_from_center),pos_flag)
    cv2.putText(img,center_text,(100,150), font, 1,(255,255,255),2)
    type_text = "left line's type: %s ,and right line's type: %s"%(left_line_type,right_line_type)
    cv2.putText(img,type_text,(100,200), font, 1,(255,255,255),2)
    color_text = "left line's color: %s ,and right line's color: %s"%(left_line_color,right_line_color)
    cv2.putText(img,color_text,(100,250), font, 1,(255,255,255),2)
    return img
