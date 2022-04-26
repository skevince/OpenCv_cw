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

def focal_length(f_p, sensor_width, image_width ):
    f_l = f_p * (sensor_width / image_width)
    print(f_l)
    return f_l


def depth(focal_length, disparity, doffs ):
    z = 174.019 * (focal_length) / disparity + doffs # baseline = 174.019
    return z

# ================================================
#
def plot(focal_length ,disparity):
    # This just plots some sample points.  Change this function to
    # plot the 3D reconstruction from the disparity map and other values
    depth_scene = 174.019 * (focal_length) / disparity + 114.291  # baseline = 174.019 5806.559, 22.2, 3088

    x = []
    y = []
    z = []
    for r in range(depth_scene.shape[0]):
        for c in range(depth_scene.shape[1]):
            x += [c]
            y += [r]
            if depth_scene[r, c] == np.max(depth_scene):
                z.append(np.NaN)
            else:
                z.append(depth_scene[r, c])

    print(len(x))
    print(len(y))
    print(len(z))


    # Plt depths
    ax = plt.axes(projection ='3d')
    ax.scatter(x, y, z, 'green')

    # Labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.savefig('myplot.pdf', bbox_inches='tight') # Can also specify an image, e.g. myplot.png
    plt.show()


# ================================================
#

if __name__ == '__main__':

    # Load left image
    filename = 'umbrellaL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    imgL = cv2.Canny(imgL, 50, 160)

    #
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # Load right image
    filename = 'umbrellaR.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.Canny(imgR, 50, 160)

    #
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

    # Get disparity map
    disparity = getDisparityMap(imgL, imgR, 64, 5)

# ===============================================
    focal_length = focal_length(5806.559, 22.2, 3088)
    doffs = 114.291
    depth_scene = depth(focal_length, disparity, doffs)
    # print(type(depth_scene))



# ===============================================
    # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.1, 2.0))

    # Show result
    cv2.imshow('Disparity', disparityImg)

    # Show 3D plot of the scene
    plot(focal_length,disparity)

    # Wait for spacebar press or escape before closing,
    # otherwise window will close without you seeing it
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

    cv2.destroyAllWindows()
