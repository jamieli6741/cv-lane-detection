import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt


def parse_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def aug_contrast(image):
    alpha = 1.2  # contrast value (1.0-3.0)  1.5
    beta = 0  # brightness value (0-100)
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return adjusted_image

def keep_yellow_white_lines(image):
    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the range for yellow color in HSV space
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    # Define the range for white color in HSV space
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([179, 30, 255])
    # Create masks based on color ranges
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    # Merge the masks for yellow and white colors
    mask = cv2.bitwise_or(mask_yellow, mask_white)
    # Extract the areas of the image that are yellow and white lines
    result = cv2.bitwise_and(image, image, mask=mask)

    return result


def canny(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5,5),0)
    frame = cv2.Canny(frame, 50, 150)

    return frame


def sobel_detect(image_gray):
    x_sobel = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=7)
    y_sobel = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=7)
    img_sobel = cv2.addWeighted(x_sobel, 0.5, y_sobel, 0.5, 0)

    return img_sobel

def get_perspective_matrix(src, dst):
    # turn front camera view into bird-eye view
    Matrix = cv2.getPerspectiveTransform(src, dst)

    return Matrix


def segment(frame, roi):
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    # Discard all pixels outside the ROI (make them black)
    blank = np.zeros_like(frame)
    mask = cv2.fillPoly(blank, roi, 255)
    masked_frame = cv2.bitwise_and(frame, mask)
    # Returns segmented image
    return masked_frame


def show_two_imgs(image1, image2, figsize=(16, 6), filename = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image1)
    ax2.imshow(image2)
    ax1.axis('off')
    ax2.axis('off')
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    plt.close()


def plot_vertical_projection(vertical_projection, peaks, filename = None):
    plt.plot(vertical_projection, color='blue', label='Vertical Projection')
    plt.scatter(peaks, vertical_projection[peaks], color='red', marker='o', label='Peaks')
    plt.title('Vertical Projection and Peaks')
    plt.xlabel('Column')
    plt.ylabel('Sum of Pixel Values')
    plt.grid(True)
    plt.legend()

    if filename is not None:
        plt.savefig(filename)
    plt.show()
    plt.close()


def plot_fitting_lines(mask, lines, filename = None):
    warp_zero = np.zeros_like(mask).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    plt.figure(figsize=(3, 5))
    for line in lines:
        y = np.linspace(0, mask.shape[0]-1, mask.shape[0])
        x = np.polyval(line, y)
        plt.plot(x, y, linewidth=2)

    plt.axis('off')
    plt.imshow(color_warp)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True, dpi=130)
    plt.show()
    plt.close()

def plot_filter_lines_opencv(mask, lines, filename=None):
    # create a black image
    color_warp = np.zeros_like(mask)
    # If the mask is single-channel, convert color_warp into a 3-channel black image
    if len(mask.shape) == 2 or mask.shape[2] == 1:
        color_warp = np.float32(color_warp)
        color_warp = cv2.cvtColor(color_warp, cv2.COLOR_GRAY2BGR)

    # Iterate through each pair of (x, y) coordinates in lines and draw line segments
    for line in lines:
        x,y = line
        for i in range(1, len(x)):
            cv2.line(color_warp, (int(x[i - 1]), int(y[i - 1])), (int(x[i]), int(y[i])), (255, 255, 0), 2)

    # If a filename is provided, save the drawn image to the specified file
    if filename is not None:
        cv2.imwrite(filename, color_warp)

    return color_warp


if __name__ == '__main__':
    config = parse_yaml("cfg/my_config.yaml")
    print(config)