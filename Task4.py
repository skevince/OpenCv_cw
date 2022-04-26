import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt


def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image

def change(x):
    k = x/100
    depth = 1 / (disparity + k)
    cv2.imshow('Depth', depth)
    #depth = 1 / (disparity + 0.91)
    print(depth.shape)
    #cv2.imshow('Depth', depth)
    filename = 'girlL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    output = cv2.GaussianBlur(imgL, (0, 0), 6)
    # cv2.imshow('girlLblur', output)
    depthImg = np.interp(depth, (depth.min(), depth.max()), (0.0, 255.0))

    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if depthImg[i][j] < 120:
                output[i][j] = imgL[i][j]

    # print(depth)
    cv2.imshow('Result', output)

def hyper(x):
    pass

if __name__ == '__main__':

    # Load left image
    filename = 'girlL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()


    # Load right image
    filename = 'girlR.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()


    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

    # Get disparity map
    disparity = getDisparityMap(imgL, imgR, 32, 47)

    # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    # Show result
    cv2.imshow('Disparity', disparity)
    cv2.createTrackbar('k', 'Disparity', 0, 1000, change)
    # cv2.createTrackbar('numDisparities', 'Disparity', 16, 512, hyper)
    # cv2.createTrackbar('blockSize', 'Disparity', 5, 255, hyper)

    # depth = 1/(disparity+0.91)
    # print(depth.shape)
    # cv2.imshow('Depth', depth)
    # filename = 'girlL.png'
    # imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #
    # output = cv2.GaussianBlur(imgL, (0, 0), 6)
    # #cv2.imshow('girlLblur', output)
    # depthImg = np.interp(depth, (depth.min(), depth.max()), (0.0, 255.0))
    #
    #
    # for i in range (depth.shape[0]):
    #     for j in range(depth.shape[1]):
    #         if depthImg[i][j] < 120 :
    #             output[i][j] = imgL[i][j]
    #
    #
    #
    # #print(depth)
    # cv2.imshow('Result',output)

    # Show 3D plot of the scene
    #plot(disparity)

    # Wait for spacebar press or escape before closing,
    # otherwise window will close without you seeing it
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break


        # numD = cv2.getTrackbarPos('numDisparities', 'Disparity')
        # bs = cv2.getTrackbarPos('blockSize', 'Disparity')
        # if numD % 16 != 0 or bs % 2 == 0:
        #     continue
        # else:
        #     disparity = getDisparityMap(imgL, imgR, numD, bs)
        #     disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    cv2.destroyAllWindows()