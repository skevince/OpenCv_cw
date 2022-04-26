import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt


def change(objects):
    pass


def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1  # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0  # Map is fixed point int with 4 fractional bits

    return disparity  # floating point image


def getDepth(disparity, k):
    d = 1 / (disparity + (k / 100))
    return d


def getDepthimg(depth, img):
    output = cv2.GaussianBlur(img, (0, 0), 6)
    depthImg = np.interp(depth, (depth.min(), depth.max()), (0.0, 255.0))
    if depth.shape == img.shape:
        for r in range(output.shape[0]):
            for c in range(output.shape[1]):
                if depthImg[r][c] < 110:
                    output[r][c] = img[r][c]
    return output


if __name__ == '__main__':

    # Load left image
    filename = 'girlL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # imgL = cv2.Canny(imgL, 70, 150)
    #
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # Load right image
    filename = 'girlR.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # imgR = cv2.Canny(imgR, 70, 150)
    #
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)

    # Get disparity map
    disparity = getDisparityMap(imgL, imgR, 16, 45)
    # # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    # Get depth
    depth = getDepth(disparity, 10)
    depthimg = getDepthimg(depth, imgL)
    # Show result
    cv2.imshow('Depth', depthimg)
    cv2.createTrackbar('k_value', 'Depth', 0, 500, change)

    while True:
        cv2.imshow('Depth', depthimg)
        # Get trackbars
        k = cv2.getTrackbarPos('k_value', 'Depth')

        if k % 1 != 0:
            continue
        else:
            disparity = getDisparityMap(imgL, imgR, 16, 45)
            depth = getDepth(disparity, k)
            depthimg = getDepthimg(depth, imgL)
            print(k)


        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

    cv2.destroyAllWindows()