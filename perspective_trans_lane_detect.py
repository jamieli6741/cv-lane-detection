import copy

import numpy as np
from scipy.signal import find_peaks
from utils import *


def finding_line(warped_mask, x_points, sliding_window_num=9, margin=15, min_pixels_threshold=50):
    height, width = warped_mask.shape
    # Get the coordinates of all non-zero pixels in the image
    nonzero_y, nonzero_x = np.nonzero(warped_mask)
    # Calculate the height of the sliding window
    sliding_window_height = height // sliding_window_num
    # Store pixel indices within each sliding window
    line_pixel_indexes = [[] for _ in range(len(x_points))]

    # Iterate through sliding windows
    for i in range(sliding_window_num):
        for idx, x_point in enumerate(x_points):
            # Determine the window's boundaries on the y-axis
            top, bottom = height - (i + 1) * sliding_window_height, height - i * sliding_window_height
            # Determine the window's boundaries on the x-axis
            left, right = x_point - margin, x_point + margin
            # Get the indices of non-zero pixels within the window
            window_pixel_indexes = ((nonzero_y >= top) & (nonzero_y < bottom) &
                                    (nonzero_x >= left) & (nonzero_x < right)).nonzero()[0]
            # Store the indices of pixels within the current window
            line_pixel_indexes[idx].append(window_pixel_indexes)
            # If there are enough pixels, update the coordinates of the sliding window center
            if len(window_pixel_indexes) > min_pixels_threshold:
                x_point = int(np.mean(nonzero_x[window_pixel_indexes]))

    # For storing the coefficients of the fitted lines
    lines = []
    # Process the pixel indices for each sliding window
    for line_pixel_index in line_pixel_indexes:
        # Merge pixel indices
        line_pixel_index = np.concatenate(line_pixel_index)
        if len(line_pixel_index) > 0:
            # Extract coordinates
            line_x, line_y = nonzero_x[line_pixel_index], nonzero_y[line_pixel_index]

            # calculate pattern's width/height ratio to see whether it is a line
            shape_ratio = (max(line_x)-min(line_x))/ (max(line_y)-min(line_y))

            if shape_ratio <= 0.5 and max(line_y) > 300:
                # Fit a polynomial curve and add the result to lines
                lines.append(np.polyfit(line_y, line_x, 2))

    return lines


def perepective_trans_detect(image, config):
    pixel_points = np.float32(config['pixel_points'])
    actual_points = np.float32(config['actual_points'])
    roi_coords = np.float32(config['roi_coords'])
    trans_matrix = get_perspective_matrix(pixel_points, actual_points)
    try:
    # if 1:
        result = image
        frame = copy.copy(image)
        # frame = aug_contrast(frame)  # does not perform well, don't use it in this project
        frame = keep_yellow_white_lines(frame)

        # use canny method
        frame = canny(frame)  # todo check canny

        # # use sobel method
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # frame = sobel_detect(frame)

        if len(roi_coords) == 0:
            height, width = frame.shape[:2]
            roi_coords = [(width / 3, height / 2), (width * 2 / 3, height / 2), (width * 3 / 4, height * 4 / 5),
                          (width / 4, height * 4 / 5)]

        roi = np.array([roi_coords], dtype=np.int32)
        frame = segment(frame, roi)

        warped_mask = cv2.warpPerspective(frame, trans_matrix, (300, 500))
        warped_mask_colored = cv2.warpPerspective(image, trans_matrix, (300, 500))  #
        # show_two_imgs(frame, warped_mask)

        # Calculate the vertical projection of pixel values
        vertical_projection = np.sum(warped_mask, axis=0)
        # Use find_peaks function to find peak points
        peaks, _ = find_peaks(vertical_projection, height=0)  # it's index

        original_lines = finding_line(warped_mask, peaks)
        # plot_vertical_projection(vertical_projection, peaks)

        # filter with vertical_projection value
        fpeaks = [index for index in peaks if vertical_projection[index] > 100]

        # # filter adjacent lines
        if len(fpeaks) > 0:
            pixel_values = vertical_projection[fpeaks]
            index_big2small = pixel_values.argsort()[::-1]
            fpeaks_v_value_big2small = np.array(fpeaks)[index_big2small]
            pixel_values_big2small = vertical_projection[fpeaks_v_value_big2small]

            filtered_peaks = [fpeaks_v_value_big2small[0]]
            count = 0
            for peak, proj_val in zip(fpeaks_v_value_big2small[1:], pixel_values_big2small[1:]):
                if all(abs(filtered_peaks-peak) >= 20):
                    filtered_peaks.append(peak)
                    count += 1
                if count >= 4:  # keep 4 candidate lines
                    break
            filtered_peaks = np.array(filtered_peaks)
            # plot_vertical_projection(vertical_projection, filtered_peaks)

            filter_lines = finding_line(warped_mask, filtered_peaks)
            # filter_lines = finding_line(warped_mask, fpeaks)

            # plot_fitting_lines(warped_mask, filter_lines)

            # further filter after polyfit
            final_line_xys = []
            for idx, line in enumerate(filter_lines):
                y = np.linspace(0, warped_mask.shape[0] - 1, warped_mask.shape[0])
                x = np.polyval(line, y)
                max_x = max(x)

                if idx == 0:
                    final_line_xys.append((x, y))
                    prev_b, prev_max_x = line[1], max_x
                else:
                    if abs(max_x - prev_max_x) < 20:
                        if abs(prev_b) > abs(line[1]):
                            final_line_xys[-1] = (x, y)
                            prev_b, prev_max_x = line[1], max_x
                    else:
                        final_line_xys.append((x, y))
                        prev_b, prev_max_x = line[1], max_x

            # # only keep two lines which are the closest to the image center
            # dist2c_left = []
            # dist2c_right = []
            # for i, line_xys in enumerate(final_line_xys):
            #     # the bird-eye view image size is set to be [500, 300], so the image center is at 150
            #     x_avg = np.mean(line_xys[0])
            #     if x_avg < 150:
            #         dist2c_left.append([i, 150-x_avg])
            #     else:
            #         dist2c_right.append([i, x_avg-150])
            # if len(dist2c_left) == 0:
            #     index_keep = np.array(dist2c_right)[np.argsort(np.array(dist2c_right)[:,1])[:2]][:,0].astype(np.int32)
            # elif len(dist2c_right) == 0:
            #     index_keep = np.array(dist2c_left)[np.argsort(np.array(dist2c_left)[:,1])[:2]][:,0].astype(np.int32)
            # else:
            #     index_keep_left = np.array(dist2c_left)[np.argsort(np.array(dist2c_left)[:,1])[0]][0].astype(np.int32)
            #     index_keep_right = np.array(dist2c_right)[np.argsort(np.array(dist2c_right)[:,1])[0]][0].astype(np.int32)
            #     index_keep = [index_keep_left, index_keep_right]
            # final_line_xys = np.array(final_line_xys)[index_keep]

            img = plot_filter_lines_opencv(warped_mask, final_line_xys)

            # Project the fitted lane lines back onto the original image
            newwarp = cv2.warpPerspective(img, np.linalg.inv(trans_matrix), (image.shape[1], image.shape[0]))
            newwarp[:newwarp.shape[0] // 2, :] = 0
            newwarp[-config['show_y']:, :] = 0  # plot from the car head
            image = image.astype(np.uint8)
            newwarp = newwarp.astype(np.uint8)
            result = cv2.addWeighted(image, 1, newwarp, 1, 0)
        return result

    except:
        return image



if __name__ == '__main__':
    # img_path = 'sample_images/00d8944b-e157478b_sample.png'
    # config = parse_yaml('cfg/my_config.yaml')
    # image = cv2.imread(img_path)
    # perepective_trans_detect(image, config)

    # detect in video stream
    video_path = '/home/jamie/Works/McMaster/my_lane_detection/Input/Input0.mp4'
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
    output_path = "Output/{}_{}.mp4".format(video_path.split('/')[-1].split('.')[0], method)
    # output_path = "final_videos/input0_curved.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while (cap.isOpened()):
        ret, frame = cap.read()
        output = perepective_trans_detect(frame, config)
        cv2.imshow("output", output)
        out.write(output)  # write to output file
        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    # The following frees up resources and closes all windows
    cap.release()
    cv2.destroyAllWindows()