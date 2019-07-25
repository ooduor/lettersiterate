#!/usr/bin/env python
import argparse
import cv2
import numpy as np
from pprint import pprint

from utils import lines_extraction, draw_lines, extract_polygons

def main(args):
    # get params
    path_to_image = args.image
    image = cv2.imread(path_to_image) #reading the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting to grayscale image
    (thresh, im_bw) = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # converting to binary image
    # invert image data using unary tilde operator
    # im_bw = ~im_bw

    # Perform opening on the thresholded image (erosion followed by dilation)
    kernel = np.ones((2,2),np.uint8)
    im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel) # cleans up random lines that appear on the page

    # extract and draw any lines from the image
    lines_mask = draw_lines(image, gray)

    # extract complete shapes likes boxes of ads and banners
    found_polygons_mask = extract_polygons(im_bw, lines_mask)

    # nullifying the mask of unwanted polygons over binary (toss images)
    # this should not only have texts, without images
    text_im_bw = cv2.bitwise_and(im_bw, im_bw, mask=found_polygons_mask)

    # initialize blank image for extracted titles
    titles_mask = np.ones(image.shape[:2], dtype="uint8") * 255
    contents_mask = np.ones(image.shape[:2], dtype="uint8") * 255

    (contours, _) = cv2.findContours(text_im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(contour)[3] for contour in contours]
    avgheight = sum(heights)/len(heights)

    # finding the larger text
    for c in contours:
        [x,y,w,h] = cv2.boundingRect(c)
        cv2.rectangle(contents_mask, (x,y), (x+w,y+h), (255, 0, 0), 1)
        if h > 2*avgheight:
            cv2.drawContours(titles_mask, [c], -1, 0, -1)
        elif h*w > 20: # remove specks on dots
            # get the biggest chunks of texts... articles!
            cv2.drawContours(contents_mask, [c], -1, 0, -1)

    pprint(heights)
    cv2.imwrite('im_bw.png', im_bw)
    cv2.imwrite('lines_mask.png', lines_mask)
    cv2.imwrite('contents_mask.png', contents_mask)
    # extract and draw any lines from the image
    lines_mask = draw_lines(contents_mask, gray)
    cv2.imwrite('contents_lines_mask.png', lines_mask)

    print('Main code {} {}'.format(args.image, args.iteras))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(prog="LettersIterate", description='Split Veritable Columns in a Newspaper Like Image')
    parser.add_argument('image', type=str, help='Path to the image file') # Required positional argument
    parser.add_argument('--iteras', type=int, help='An optional integer argument') # Optional argument
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    args = parser.parse_args()

    # execute only if run as the entry point into the program
    main(args)