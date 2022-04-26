import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

def change(x):
    # Change brightess by adding a value to every pixel.
    # That is, the value sent from the TrackBar
    pass

# ================================================
#
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image
# ================================================

# ================================================
def plot(disparity):
    # This just plots some sample points.  Change this function to
    # plot the 3D reconstruction from the disparity map and other values

    baseline = 174.019
    f = 5806.559
    doffs = 114.291
    # scale the doffs
    print(doffs)

    x = []
    y = []
    z = []

    for r in range(disparity.shape[0]):
        for c in range(disparity.shape[1]):
            depth = baseline * f / (disparity[r][c] + doffs)
            if (depth > 7500): 
                continue
            else:
                y += [r/f * depth]
                x += [c/f * depth]
                z += [depth]

    # Plt depths
    ax = plt.axes(projection ='3d')
    ax.scatter(x, z, y, 'green',s=1)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    # ax.view_init(45, 45)
    # ax.view_init(20, -120)
    # ax.view_init(90, -90)
    ax.view_init(180, 0)
    # Labels
    plt.savefig('myplot.pdf', bbox_inches='tight') # Can also specify an image, e.g. myplot.png
    plt.show()


# ================================================
#
if __name__ == '__main__':

    # Load left image
    filename = 'umbrellaL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()
    cv2.imwrite("umbrellaL_greyscale.png",imgL)

    # Load right image
    filename = 'umbrellaR.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("umbrellaR_greyscale.png", imgR)
    #
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    cv2.namedWindow('Disparity_greyscale', cv2.WINDOW_NORMAL)
    # Get disparity map
    disparity = getDisparityMap(imgL, imgR, 16, 5)
    # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    cv2.createTrackbar("numDisparities", "Disparity_greyscale", 16, 512, change)
    cv2.createTrackbar("blockSize", "Disparity_greyscale", 5, 125, change)

    # Wait for space bar press or escape before closing,
    # otherwise window will close without you seeing it
    while True:
        cv2.imshow('Disparity_greyscale', disparityImg)
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break
        # get current positions of my track bar
        nums = cv2.getTrackbarPos("numDisparities", "Disparity_greyscale")
        blocks = cv2.getTrackbarPos("blockSize", "Disparity_greyscale")

        if nums % 16 != 0 or blocks % 2 == 10:
            nums = int(nums / 16) * 16
            blocks = blocks - 1
        else:
            # Get disparity map
            disparity = getDisparityMap(imgL, imgR, nums, blocks)
            # Normalise for display
            disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    # Show result
    cv2.imshow('Disparity', disparityImg)
    cv2.imwrite("disparity_greyscale.png", disparityImg)

    # 边缘检测
    edgeL = cv2.Canny(imgL, 70, 150)
    edgeR = cv2.Canny(imgR, 70, 150)

    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    # Get disparity map
    disparity = getDisparityMap(edgeL, edgeR, 16, 5)
    # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    cv2.createTrackbar("numDisparities", "Disparity", 16, 512, change)
    cv2.createTrackbar("blockSize", "Disparity", 5, 125, change)


    # Wait for space bar press or escape before closing,
    # otherwise window will close without you seeing it
    while True:
        cv2.imshow('Disparity', disparityImg)
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break
        # get current positions of my track bar
        nums = cv2.getTrackbarPos("numDisparities", "Disparity")
        blocks = cv2.getTrackbarPos("blockSize", "Disparity")

        if nums % 16 != 0 or blocks % 2 == 10:
            nums = int(nums / 16) * 16
            blocks = blocks - 1
        else:
            # Get disparity map
            disparity = getDisparityMap(edgeL, edgeR, nums, blocks)
            # Normalise for display
            disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    # Show result
    cv2.imshow('Disparity', disparityImg)
    cv2.imwrite("disparity_edge.png", disparityImg)
    # Show 3D plot of the scene
    plot(disparity)
    cv2.destroyAllWindows()

