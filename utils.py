import cv2
from typing import List, Dict
import numpy as np

def lines_extraction(gray: List[int]) -> List[int]:
    """
    This function extracts the lines from the binary image. Cleaning process.
    """
    edges = cv2.Canny(gray, 50, 150)
    rho_res = 1 # [pixels]
    theta_res = np.pi/180 # [radians]
    threshold = 50 # number of votes to be considered a line
    min_line_length = 5
    max_line_gap = 50
    lines = cv2.HoughLinesP(edges, rho_res, theta_res, threshold, min_line_length, max_line_gap)

    return lines

def draw_lines(image, gray):
    lines = lines_extraction(gray) # line extraction

    # create blank masks with same dimensions as the original image
    lines_mask = np.ones(image.shape[:2], dtype="uint8") * 255
    try:
        for line in lines:
            """
            drawing extracted lines on mask
            """
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_mask, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)
    except TypeError:
        pass

    return lines_mask

def extract_polygons(im_bw, lines_mask):
    """
    Returns contigious shapes of polygon in the page
    """
    (contours, _) = cv2.findContours(im_bw, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    avgArea = sum(areas)/len(areas)
    for c in contours:
        if cv2.contourArea(c)>200*avgArea:
            cv2.drawContours(lines_mask, [c], -1, 0, -1)

    return lines_mask

def column_summaries(image, rlsa_mask):
    (for_avgs_contours, _) = cv2.findContours(~rlsa_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for_avgs_contours_mask = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image
    contents_sum_list = []
    contents_x_list = [] # to get the left-most content box
    contents_length = 0
    for idx, contour in enumerate(for_avgs_contours):
        [x, y, w, h] = cv2.boundingRect(contour)
        # apply some heuristic to different other stranger things masquerading as titles
        if w*h > 1500: # remove tiny contours the dirtify the image
            cv2.drawContours(for_avgs_contours_mask, [contour], -1, 0, -1)
            cv2.rectangle(for_avgs_contours_mask, (x,y), (x+w,y+h), (255, 0, 0), 5)
            cv2.putText(for_avgs_contours_mask, "#{},x{},y{},w{}".format(idx, x, y, w), cv2.boundingRect(contour)[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [0, 0, 255], 2) # [B, G, R]
            contents_sum_list.append(w)
            contents_x_list.append(x)
            contents_length += 1

    return contents_sum_list, contents_x_list, for_avgs_contours_mask

def determine_precedence(contour, cols, avgwidth, leftmost_x, m_height):
    """
    Sort contours by distance from...
    https://stackoverflow.com/questions/39403183/python-opencv-sorting-contours
    """
    tolerance_factor = 10
    [x,y,w,h] = cv2.boundingRect(contour)
    i = 1
    col_height = 0
    x_adjustment = avgwidth
    col_loc = None
    while i <= cols:
        # for the first loop only, offset with beginning of first title
        if i == 1:
            avgwidth = avgwidth + leftmost_x

        if x <= avgwidth:
            col_loc = ((x / tolerance_factor) * tolerance_factor) * i + y + col_height
            break
        i = i + 1
        avgwidth = x_adjustment*i
        col_height = col_height+(m_height*2)

    if col_loc is None: # if wasn't within any of the columns put it in the last one atleast
        col_loc = ((x / tolerance_factor) * tolerance_factor) * cols + y + col_height

    return col_loc

def redraw_titles(image, contours):
    """
    Redraw the titles successfully extractedon a white mask
    """
    clear_titles_mask = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image

    for idx, contour in enumerate(contours):
        [x, y, w, h] = cv2.boundingRect(contour)
        # cv2.drawContours(clear_titles_mask, [contour], -1, 0, -1)
        # cv2.rectangle(clear_titles_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
        titles = image[y: y+h, x: x+w]
        clear_titles_mask[y: y+h, x: x+w] = titles # copied titles contour onto the blank image
        image[y: y+h, x: x+w] = 255 # nullified the titles contour on original image
        # cv2.putText(clear_titles_mask, "#{},x{},y{},w{},h{}".format(idx, x, y, w, h), cv2.boundingRect(contour)[:2], cv2.FONT_HERSHEY_PLAIN, 1.50, [255, 0, 0], 2) # [B, G, R]

    return clear_titles_mask

def redraw_contents(image, contours):
    """
    Redraw the titles successfully extractedon a white mask
    """
    clear_contents_mask = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image

    # for idx, contour in enumerate(contours):
    for idx in range(len(contours)):
        [x, y, w, h] = cv2.boundingRect(contours[idx])
        # cv2.drawContours(clear_contents_mask, [c], -1, 0, -1)
        # cv2.rectangle(clear_contents_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
        contents = image[y: y+h, x: x+w]
        clear_contents_mask[y: y+h, x: x+w] = contents # copied contents contour onto the blank image
        # image[y: y+h, x: x+w] = 255 # nullified the contents contour on original image
        # cv2.putText(clear_contents_mask, "#{},x{},y{}".format(idx, x, y), cv2.boundingRect(contours[idx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 153, 255], 2) # [B, G, R]

    # cv2.imwrite('clear_contents_mask.png', clear_contents_mask) # debug remove
    return clear_contents_mask

def draw_columns(leftmost_x, trimmed_mean, total_columns, clear_titles_mask):
    counter = 1
    x = leftmost_x + trimmed_mean
    while counter <= total_columns:
        cv2.line(clear_titles_mask, (x, 0), (x, 2000), (0,255,0), 2)
        counter+=1
        x+=trimmed_mean