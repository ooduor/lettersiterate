#!/usr/bin/env python
import os
import sys
import glob
from pprint import pprint
import math
import argparse
import cv2
import imutils
import numpy as np
from scipy import stats
from pythonRLSA import rlsa

from utils import lines_extraction, draw_lines, extract_polygons, \
    column_summaries, determine_precedence, redraw_titles, redraw_contents, \
    draw_columns

def main(args):
    # get params
    path_to_image = args.image
    empty_output = args.empty

    # check if file exists here and exist if not
    try:
        f = open(path_to_image)
        f.close()
    except FileNotFoundError:
        print('Image given does not exist')
        sys.exit(0)

    # create out dir
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'output')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # standardize size of the images maintaining aspect ratio
    if empty_output:
        files = glob.glob('{}/*'.format(final_directory))
        for f in files:
            os.remove(f)

    image = cv2.imread(path_to_image) #reading the image

    image_width = image.shape[1]
    if image_width != 2048:
        image = imutils.resize(image, width=2048)
        cv2.imwrite('image.png' , image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting to grayscale image
    (thresh, im_bw) = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # converting to binary image
    # invert image data using unary tilde operator
    # im_bw = ~im_bw

    # Noise removal step - Perform opening on the thresholded image (erosion followed by dilation)
    kernel = np.ones((2,2),np.uint8) # kernel noise size (2,2)
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

    # helps further detach titles if necessary. This step can be removed
    # titles_mask = cv2.erode(titles_mask, kernel, iterations = 1)
    m_height, m_width = titles_mask.shape # get image dimensions, height and width

    # run length smoothing algorithms for vertical and lateral conjoining of pixels
    value = max(math.ceil(m_height/100), math.ceil(m_width/100))+1
    rlsa_titles_mask = rlsa.rlsa(titles_mask, True, False, value) #rlsa application
    rlsa_titles_mask_for_final = rlsa_titles_mask

    value = max(math.ceil(m_height/100),math.ceil(m_width/100))+20
    rlsa_contents_mask = rlsa.rlsa(contents_mask, False, True, value) #rlsa application
    rlsa_contents_mask_for_avg_width = rlsa_contents_mask

    # get avg properties of columns
    contents_sum_list, contents_x_list, for_avgs_contours_mask = column_summaries(image, rlsa_contents_mask_for_avg_width)
    cv2.imwrite(os.path.join(final_directory, 'for_avgs_contours_mask.png'), for_avgs_contours_mask) # debug remove
    trimmed_mean = int(stats.trim_mean(contents_sum_list, 0.1)) # trimmed mean
    leftmost_x = min(contents_x_list)

    threshold = 1500 # remove tiny contours that dirtify the image

    ### titles work
    (contours, _) = cv2.findContours(~rlsa_titles_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # apply some heuristic to differentiate other stranger things masquerading as titles
    nt_contours = [contour for contour in contours if cv2.boundingRect(contour)[2]*cv2.boundingRect(contour)[3] > threshold]

    total_columns = int(image.shape[1]/trimmed_mean)
    print(total_columns)
    contours = sorted(nt_contours, key=lambda contour:determine_precedence(contour, total_columns, trimmed_mean, leftmost_x, m_height))
    clear_titles_mask = redraw_titles(image, contours)

    # draw_columns(leftmost_x, trimmed_mean, total_columns, clear_titles_mask)
    cv2.imwrite(os.path.join(final_directory, 'clear_titles_mask.png'), clear_titles_mask) # debug remove

    ### contents work
    (contours, _) = cv2.findContours(~rlsa_contents_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # apply some heuristic to different other stranger things masquerading as titles
    nt_contours = [contour for contour in contours if cv2.boundingRect(contour)[2]*cv2.boundingRect(contour)[3] > threshold]

    contents_contours = sorted(nt_contours, key=lambda contour:determine_precedence(contour, total_columns, trimmed_mean, leftmost_x, m_height))
    clear_contents_mask = redraw_contents(image, contents_contours)

    # start printing individual articles based on titles! The final act
    (contours, _) = cv2.findContours(~rlsa_titles_mask_for_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # apply some heuristic to different other stranger things masquerading as titles
    nt_contours = [contour for contour in contours if cv2.boundingRect(contour)[2]*cv2.boundingRect(contour)[3] > threshold]

    contours = sorted(nt_contours, key=lambda contour:determine_precedence(contour, total_columns, trimmed_mean, leftmost_x, m_height))

    article_complete = False
    title_lines_count = 0
    article_mask = np.ones(image.shape, dtype="uint8") * 255 # blank layer image for one article
    # for idx, contour in enumerate(contours):
    for idx, (_curr, _next) in enumerate(zip(contours[::],contours[1::])):
        # https://www.quora.com/How-do-I-iterate-through-a-list-in-python-while-comparing-the-values-at-adjacent-indices/answer/Jignasha-Patel-14
        if article_complete:
            article_mask = np.ones(image.shape, dtype="uint8") * 255 # blank layer image for antother separate article
        [cx, cy, cw, ch] = cv2.boundingRect(_curr)
        [nx, ny, nw, nh] = cv2.boundingRect(_next)
        if (ny-cy) > (nh+ch)*2: # next is greater than current...
            print('Big Gap! {}'.format(idx))

            # loop through contents and insert any valid ones in this gap
            for idxx in range(len(contents_contours)):
                [x, y, w, h] = cv2.boundingRect(contents_contours[idxx])
                # search_area_rect = cv2.rectangle(clear_contents_mask,(cx,cy),(x+w,y+h),(0,0,255),thickness=3,shift=0)
                dist = cv2.pointPolygonTest(contents_contours[idxx],(x,y), False)
                # https://stackoverflow.com/a/50670359/754432
                if cy < y and cx-10 < x and x < (cx+w): # less than because it appears above
                    # check but not greater than the next title!!
                    if y > ny: # or next is another column
                        continue
                    # cv2.drawContours(clear_contents_mask, [c], -1, 0, -1)
                    # cv2.rectangle(clear_contents_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
                    contents = clear_contents_mask[y: y+h, x: x+w]
                    article_mask[y: y+h, x: x+w] = contents # copied title contour onto the blank image
                    image[y: y+h, x: x+w] = 255 # nullified the title contour on original image
                    # cv2.putText(clear_contents_mask, "#{},x{},y{}".format(idxx, x, y), cv2.boundingRect(contours[idxx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 153, 255], 2) # [B, G, R]

                #  continued article
                if x < (cx+cw) and y < ny and cx < x: # covered by length of title | is anothe column
                    # cv2.drawContours(clear_contents_mask, [c], -1, 0, -1)
                    # cv2.rectangle(clear_contents_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
                    contents = clear_contents_mask[y: y+h, x: x+w]
                    article_mask[y: y+h, x: x+w] = contents # copied title contour onto the blank image
                    # cv2.putText(clear_contents_mask, "#{},x{},y{}".format(idxx, x, y), cv2.boundingRect(contours[idxx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 153, 255], 2) # [B, G, R]

                    # check but not greater than the next title!!
                    if y > ny: # or next is another column
                        continue

            article_title_p = clear_titles_mask[cy: cy+ch, cx: cx+cw]
            article_mask[cy: cy+ch, cx: cx+cw] = article_title_p # copied title contour onto the blank image
            image[cy: cy+ch, cx: cx+cw] = 255 # nullified the title contour on original image

            cv2.imwrite(os.path.join(final_directory, 'article_{}big.png'.format(idx)), article_mask)
            article_complete = True
        elif (ny-cy) < (nh+ch)*2 and (ny-cy) > 0 and nx <= cx+10: # next if not greater... but just small|
            print('Small Gap! {}'.format(idx))

            # handle special cases like end of the page
            if len(contours) == (idx+2): # we are on last article, it's always greater by 2 instead of one. Nkt!
                # loop through contents and insert any valid ones in this gap
                for idxx in range(len(contents_contours)):
                    [x, y, w, h] = cv2.boundingRect(contents_contours[idxx])
                    if cy-ch < y and (cy+ch) < (y+h) and (cx+cw) < (x+w): # more than because it appears above but not too above
                        # cv2.drawContours(clear_contents_mask, [c], -1, 0, -1)
                        # cv2.rectangle(clear_contents_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
                        contents = clear_contents_mask[y: y+h, x: x+w]
                        article_mask[y: y+h, x: x+w] = contents # copied title contour onto the blank image
                        image[y: y+h, x: x+w] = 255 # nullified the title contour on original image
                        # cv2.putText(clear_contents_mask, "#{},x{},y{}".format(idxx, x, y), cv2.boundingRect(contours[idxx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 153, 255], 2) # [B, G, R]

                        # check that it does not encounter new title in next column
                        if y > ny:
                            break

                article_title_p = clear_titles_mask[cy: cy+ch, cx: cx+cw]
                article_mask[cy: cy+ch, cx: cx+cw] = article_title_p # copied title contour onto the blank image
                image[cy: cy+ch, cx: cx+cw] = 255 # nullified the title contour on original image

                article_title_p = clear_titles_mask[ny: ny+nh, nx: nx+nw]
                article_mask[ny: ny+nh, nx: nx+nw] = article_title_p # copied title contour onto the blank image
                image[ny: ny+nh, nx: nx+nw] = 255 # nullified the title contour on original image

                cv2.imwrite(os.path.join(final_directory, 'article_{}.png'.format(idx)), article_mask)

            article_title_p = clear_titles_mask[cy: cy+ch, cx: cx+cw]
            article_mask[cy: cy+ch, cx: cx+cw] = article_title_p # copied title contour onto the blank image
            image[cy: cy+ch, cx: cx+cw] = 255 # nullified the title contour on original image

            article_complete = False

        elif (ny-cy) < (nh+ch)*2 and (ny-cy) < 0: # next is not greater... must be invalid
            print('Invalid Gap! {}'.format(idx))
            # loop through contents and insert any valid ones in this gap
            for idxx in range(len(contents_contours)):
                [x, y, w, h] = cv2.boundingRect(contents_contours[idxx])
                if cy-ch < y and cx-10 < x: # more than because it appears above but not too above
                    # cv2.drawContours(clear_contents_mask, [c], -1, 0, -1)
                    # cv2.rectangle(clear_contents_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
                    contents = clear_contents_mask[y: y+h, x: x+w]
                    article_mask[y: y+h, x: x+w] = contents # copied title contour onto the blank image
                    image[y: y+h, x: x+w] = 255 # nullified the title contour on original image
                    # cv2.putText(clear_contents_mask, "#{},x{},y{}".format(idxx, x, y), cv2.boundingRect(contours[idxx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 153, 255], 2) # [B, G, R]

                    # check that it does not encounter new title in next column
                    if y > ny:
                        break

            article_title_p = clear_titles_mask[cy: cy+ch, cx: cx+cw]
            article_mask[cy: cy+ch, cx: cx+cw] = article_title_p # copied title contour onto the blank image
            image[cy: cy+ch, cx: cx+cw] = 255 # nullified the title contour on original image

            cv2.imwrite(os.path.join(final_directory, 'article_{}invalid.png'.format(idx)), article_mask)
            article_complete = True

        else: # must be first one with next invalid...
            print('Invalid First Gap! {}'.format(idx))
            # loop through contents and insert any valid ones in this gap
            for idxx in range(len(contents_contours)):
                [x, y, w, h] = cv2.boundingRect(contents_contours[idxx])
                if cy-ch < y and cx-10 < x: # more than because it appears above but not too above
                    # cv2.drawContours(clear_contents_mask, [c], -1, 0, -1)
                    # cv2.rectangle(clear_contents_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
                    contents = clear_contents_mask[y: y+h, x: x+w]
                    article_mask[y: y+h, x: x+w] = contents # copied title contour onto the blank image
                    image[y: y+h, x: x+w] = 255 # nullified the title contour on original image
                    # cv2.putText(clear_contents_mask, "#{},x{},y{}".format(idxx, x, y), cv2.boundingRect(contours[idxx])[:2], cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 153, 255], 2) # [B, G, R]

                    # check that it does not encounter new title in next column
                    if y > ny or nx-10 > x: # its lower in the page || the next title even with 10px offset is still larger... then we are tresspassing
                        break

            article_title_p = clear_titles_mask[cy: cy+ch, cx: cx+cw]
            article_mask[cy: cy+ch, cx: cx+cw] = article_title_p # copied title contour onto the blank image
            image[cy: cy+ch, cx: cx+cw] = 255 # nullified the title contour on original image

            cv2.imwrite(os.path.join(final_directory, 'article_{}invalidfirst.png'.format(idx)), article_mask)
            article_complete = True

    print('Main code {} {}'.format(args.image, args.empty))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(prog="LettersIterate", description='Split Veritable Columns in a Newspaper Like Image')
    parser.add_argument('image', type=str, help='Path to the image file') # Required positional argument
    parser.add_argument('--empty', type=bool, help='An optional boolean argument to empty output folder before each processing', default=True) # Optional argument
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    args = parser.parse_args()

    # execute only if run as the entry point into the program
    main(args)