import cv2
import numpy as np
from perspective_trans_lane_detect import perepective_trans_detect
from utils import *


def calculate_lines(frame, lines):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    # Loops through every detected line
    slopes = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    slope_avg = np.mean(slopes)

    for line in lines:
        # Reshapes line from 2D array to 1D array
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]

        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
            
    left_avg = np.average(left,axis=0)
    right_avg = np.average(right,axis=0)

    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)

    return np.array([left_line, right_line])


def calculate_coordinates(frame, parameters):
    slope, intercept = parameters
    # Sets initial y-coordinate as height from car head
    y1 = frame.shape[0]-config['show_y']
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1-200) # int(y1 - 150)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def visualize_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    frame_lines = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    height, width, _ = frame_lines.shape
    # Checks if any lines are detected
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv2.line(frame_lines, (x1,y1), (x2,y2), (0,255,0), 10)
    return frame_lines


def hough_detect(frame, config):
    # Image augmentation
    frame_aug = keep_yellow_white_lines(frame)
    frame_canny = canny(frame_aug)

    # Apply Segment (ROI)
    roi_coords = config['roi_coords']
    if len(roi_coords) == 0:
        height, width = frame.shape[:2]
        roi_coords = [(width/3, height/2), (width*2/3, height/2), (width*3/4, height*4/5), (width/4, height*4/5)]

    roi = np.array([roi_coords], dtype=np.int32)
    frame_roi = segment(frame_canny, roi)

    # Generate Hough lines using HoughLinesP with minLineLength 100 and maLineGap 50
    lines = cv2.HoughLinesP(frame_roi, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
    try:
        lines = calculate_lines(frame_roi, lines)
        # Visualizes the lines
        lines_visualize = visualize_lines(frame, lines)
        output = cv2.addWeighted(frame, 0.9, lines_visualize, 1, 1)
        return output
    except:
        return frame

def detect_lanes(frame, config):
    method = config['lane_detect_mode']
    assert method in ['hough', 'perspective']
    if method == 'hough':
        # Method1: Hough Transformation
        output = hough_detect(frame, config)
    else:
        # Method2: Perspective transformation
        output = perepective_trans_detect(frame, config)

    return output


if __name__ == '__main__':
    # # detect single image
    # frame = cv2.imread('sample_images/00d8944b-e157478b_sample.png')
    # config = parse_yaml('cfg/my_config.yaml')
    # output = detect_lanes(frame, config)  # perspective , hough
    # # Opens a new window and displays the output frame
    # cv2.imshow("output", output)
    # cv2.waitKey(0)

    # detect in video stream
    video_path = "/home/jamie/Works/McMaster/my_lane_detection/Input/00d8944b-e157478b.mov"
    # video_path = "Input/00d8944b-e157478b.mov"
    config = parse_yaml('cfg/my_config.yaml')
    method = config['lane_detect_mode']

    cap = cv2.VideoCapture(video_path)
    # Determine output video specifications
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
    output_path = "final_videos/curve.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while (cap.isOpened()):
        ret, frame = cap.read()
        output = detect_lanes(frame, config)

        # output_1 = detect_lanes(frame, method='hough', roi_coords=roi_coords)
        # output_2 = detect_lanes(frame, method='perspective', roi_coords=roi_coords)
        # output = np.vstack((output_1, output_2))
        # new_width = 800
        # original_height, original_width = output.shape[:2]
        # new_height = int((new_width / original_width) * original_height)
        # output = cv2.resize(output, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Opens a new window and displays the output frame
        # cv2.imshow("output", output)
        cv2.imshow("input", frame)
        cv2.waitKey(0)
        # out.write(output)  # write to output file
        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    # The following frees up resources and closes all windows
    cap.release()
    cv2.destroyAllWindows()