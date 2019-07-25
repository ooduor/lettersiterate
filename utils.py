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