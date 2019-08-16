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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting to grayscale image
    (thresh, im_bw) = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # converting to binary image
    # invert image data using unary tilde operator
    # im_bw = ~im_bw

    # Noise removal step - Perform opening on the thresholded image (erosion followed by dilation)
    kernel = np.ones((2,2),np.uint8) # kernel noise size (2,2)
    im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel) # cleans up random lines that appear on the page
    cv2.imwrite('im_bw.png', ~im_bw)

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

    title_widths = []
    content_widths = []
    # finding the larger text
    for c in contours:
        [x,y,w,h] = cv2.boundingRect(c)
        cv2.rectangle(contents_mask, (x,y), (x+w,y+h), (255, 0, 0), 1)
        if h > 2*avgheight:
            cv2.drawContours(titles_mask, [c], -1, 0, -1)
            title_widths.append(w)
        elif h*w > 20: # remove specks on dots
            # get the biggest chunks of texts... articles!
            cv2.drawContours(contents_mask, [c], -1, 0, -1)
            content_widths.append(w)

    # helps further detach titles if necessary. This step can be removed
    # titles_mask = cv2.erode(titles_mask, kernel, iterations = 1)
    m_height, m_width = titles_mask.shape # get image dimensions, height and width

    # make 2D Image mask of proto-original image for cutting contents
    image_mask = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image
    image_mask[0: m_height, 0: m_width] = image[0: m_height, 0: m_width]

    # run length smoothing algorithms for vertical and lateral conjoining of pixels
    value = math.ceil(sum(title_widths)/len(title_widths))*2
    print('RLSA Title Value', value)
    rlsa_titles_mask = rlsa.rlsa(titles_mask, True, False, value) #rlsa application
    rlsa_titles_mask_for_final = rlsa_titles_mask
    cv2.imwrite(os.path.join(final_directory, 'rlsa_titles_mask.png'), rlsa_titles_mask) # debug remove

    value = math.ceil(sum(content_widths)/len(content_widths))*3
    print('RLSA Content Value', value)
    rlsa_contents_mask = rlsa.rlsa(contents_mask, False, True, value) #rlsa application
    rlsa_contents_mask_for_avg_width = rlsa_contents_mask
    cv2.imwrite(os.path.join(final_directory, 'rlsa_contents_mask.png'), rlsa_contents_mask) # debug remove

    # get avg properties of columns
    contents_sum_list, contents_x_list, for_avgs_contours_mask = column_summaries(image, rlsa_contents_mask_for_avg_width)
    cv2.imwrite(os.path.join(final_directory, 'for_avgs_contours_mask.png'), for_avgs_contours_mask) # debug remove
    trimmed_mean = int(stats.trim_mean(contents_sum_list, 0.1)) # trimmed mean
    leftmost_x = min(contents_x_list)

    threshold = 2500 # remove tiny contours that dirtify the image
    ### titles work
    (contours, _) = cv2.findContours(~rlsa_titles_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # apply some heuristic to differentiate other stranger things masquerading as titles
    nt_contours = [contour for contour in contours if cv2.boundingRect(contour)[2]*cv2.boundingRect(contour)[3] > threshold]

    total_columns = int(image.shape[1]/trimmed_mean)
    contours = sorted(nt_contours, key=lambda contour:determine_precedence(contour, total_columns, trimmed_mean, leftmost_x, m_height))
    clear_titles_mask = redraw_titles(image, contours)

    draw_columns(leftmost_x, trimmed_mean, total_columns, clear_titles_mask)
    cv2.imwrite(os.path.join(final_directory, 'clear_titles_mask.png'), clear_titles_mask) # debug remove

    ### contents work
    (contours, _) = cv2.findContours(~rlsa_contents_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # apply some heuristic to different other stranger things masquerading as titles
    nt_contours = [contour for contour in contours if cv2.boundingRect(contour)[2]*cv2.boundingRect(contour)[3] > threshold]

    contents_contours = sorted(nt_contours, key=lambda contour:determine_precedence(contour, total_columns, trimmed_mean, leftmost_x, m_height))
    clear_contents_mask = redraw_contents(image_mask, contents_contours)
    draw_columns(leftmost_x, trimmed_mean, total_columns, clear_contents_mask)
    cv2.imwrite(os.path.join(final_directory, 'clear_contents_mask.png'), clear_contents_mask)

    # start printing individual articles based on titles! The final act
    (contours, _) = cv2.findContours(~rlsa_titles_mask_for_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # apply some heuristic to different other stranger things masquerading as titles
    nt_contours = [contour for contour in contours if cv2.boundingRect(contour)[2]*cv2.boundingRect(contour)[3] > threshold]

    contours = sorted(nt_contours, key=lambda contour:determine_precedence(contour, total_columns, trimmed_mean, leftmost_x, m_height))

    article_complete = False
    column_complete = False
    title_lines_count = 0
    col_no = 1 # initialize column number
    avgwidth = trimmed_mean + leftmost_x # offset with beginning of first title
    x_adjustment = trimmed_mean
    title_came_up = True
    title_count = len(contours)
    ct_widths = []
    print('Avg Width', trimmed_mean, "Total Columns", total_columns, "Leftmost", leftmost_x)
    article_mask = np.ones(image.shape, dtype="uint8") * 255 # blank layer image for one article
    # for idx, contour in enumerate(contours):
    for idx, (_curr, _next) in enumerate(zip(contours[::],contours[1::])):
        # https://www.quora.com/How-do-I-iterate-through-a-list-in-python-while-comparing-the-values-at-adjacent-indices/answer/Jignasha-Patel-14
        if article_complete:
            article_mask = np.ones(image.shape, dtype="uint8") * 255 # blank layer image for antother separate article
        [cx, cy, cw, ch] = cv2.boundingRect(_curr)
        [nx, ny, nw, nh] = cv2.boundingRect(_next)

        column_complete = False # reset column_complete flag for complete articles
        column_end = False # reset column_end flag for end of column
        barrier_title_width = nx + nw # initial width to block further content extraction
        barrier_title_height = ny # initial width to block further content extraction
        ct_height = cy+ch # title height in this column
        # iterate all titles why they are not longer than the current title
        print(f"COL_NO +++++++++++++++++++ {col_no} idx {idx}")
        while col_no <= total_columns:

            # for the first loop only, offset with beginning of first title
            if col_no == 1:
                left_boundary = 0

            print(f"@@@@@@@@ {idx} @@@@@@@@ {len(contours)}")
            print(f"({cy} < {ny} and {ny-(nh*3)} < {cy} and {nx} < {avgwidth})")
            ct_widths.append(cx+cw)
            ct_width = max(ct_widths) # adjust to get longest title width if multiple line title :)

            # dont proceed any further if the next title is right below it on same column
            # continue to next title
            # current and next have to be within the same column
            # detect last article in the columns
            if (idx+2) == title_count:
                print("Going in for last one #{}".format(idx))
                title_came_up = False
            elif (cy < ny and ny-(nh*3) < cy and nx < avgwidth):
                title_came_up = True
                break
            else:
                print("Going in for #{}".format(idx))
                print(f"({cy} < {ny} and {ny-(nh*3)} < {cy} and {nx} < {avgwidth})")
                title_came_up = False

            for tidx, tcontour in enumerate(contours):
                [tx, ty, tw, th] = cv2.boundingRect(tcontour)

                if tidx > idx:
                    # only consider titles greater than current but within this column
                    # and below current title
                    if tx > left_boundary and tx < avgwidth and ty > ct_height:
                        print(f"##{tidx} > #{idx} and {tx} > {left_boundary} and {tx} < {avgwidth} and {ty} > {ct_height}")
                        # the first title encountered that begins in this column becomes the new barrier, ignore any other titles in the column
                        # this will be the new barrier for any content starting (x) less than the barrier length...
                        # Grab any content withing this boundary
                        barrier_title_x = tx # initial x of the blocking title
                        barrier_title_width = tx + tw # initial width to block further content extraction
                        barrier_title_height = ty # initial width to block further content extraction
                        print(f"{barrier_title_width} barrier_title_width, {barrier_title_height} barrier_title_height ")
                        # loop through contents within these boundaries and insert them to the canvas
                        for content_idx, content_contour in enumerate(contents_contours):
                            [x, y, w, h] = cv2.boundingRect(content_contour)
                            # length -50 is to be safe sometimes the content cut maybe infringe onto the next title
                            if x >= left_boundary and x < avgwidth and (y+h)-50 < barrier_title_height and y+50 > ct_height:
                                print(f"{x} >= {left_boundary} and {x} < {avgwidth} and {(y+h)-50} < {barrier_title_height} and {y} > {ct_height}")
                                article_mask = cutouts(article_mask, clear_contents_mask, content_contour)
                                cv2.putText(article_mask, "#{},x{},y{},w{},h{}".format(content_idx, x, y, w, h), cv2.boundingRect(content_contour)[:2], cv2.FONT_HERSHEY_PLAIN, 1.50, [255, 0, 0], 2) #
                                column_complete = True # set brakes to not go lower since we have encountered a title if we are here
                                column_end = False

                            # print if it is the last one. Sigh!
                            if x >= left_boundary and x < avgwidth and (idx+2) == title_count and y+50 > ct_height:
                                article_mask = cutouts(article_mask, clear_contents_mask, content_contour)
                                cv2.putText(article_mask, "#{},x{},y{},w{},h{}".format(content_idx, x, y, w, h), cv2.boundingRect(content_contour)[:2], cv2.FONT_HERSHEY_PLAIN, 1.50, [255, 0, 0], 2) #
                                column_complete = True # set brakes to not go lower since we have encountered a title if we are here
                                column_end = False

                        if column_complete:
                            break

                    # if there is no title starting within this boundaries the use the last standing boundaries to gather content
                    else:
                        # detect end of column first and write all contents
                        if tx > ct_width and tidx == idx+1:
                            # loop through contents within these boundaries and insert them to the canvas
                            for content_idx, content_contour in enumerate(contents_contours):
                                [x, y, w, h] = cv2.boundingRect(content_contour)
                                print(f"============== {x} >= {cx} and {x} < {ct_width} and {y+50} > {ct_height}")
                                if (x >= cx or x >= left_boundary) and x < ct_width and y+50 > ct_height: # +50 is for those contents again that cut out title
                                    article_mask = cutouts(article_mask, clear_contents_mask, content_contour)
                                    cv2.putText(article_mask, "#{},x{},y{},w{},h{}".format(content_idx, x, y, w, h), cv2.boundingRect(content_contour)[:2], cv2.FONT_HERSHEY_PLAIN, 1.50, [255, 0, 0], 2) #
                                    column_complete = True
                                    column_end = True
                        elif tidx > idx and tx > left_boundary and tx > avgwidth and ty > ct_height: # two part comment with first part full height
                            # loop through contents within these boundaries and insert them to the canvas
                            for content_idx, content_contour in enumerate(contents_contours):
                                [x, y, w, h] = cv2.boundingRect(content_contour)
                                if x >= left_boundary and x < avgwidth and y+50 > ct_height:
                                    print(f"col_no {col_no} ##{tidx} ###{content_idx}||||||||||||====== {x} >= {cx} and {x} < {ct_width} and {y} > {ct_height}")
                                    article_mask = cutouts(article_mask, clear_contents_mask, content_contour)
                                    cv2.putText(article_mask, "#{},x{},y{},w{},h{}".format(content_idx, x, y, w, h), cv2.boundingRect(content_contour)[:2], cv2.FONT_HERSHEY_PLAIN, 1.50, [255, 0, 0], 2) #
                                    column_complete = True
                                    column_end = True
                        else:
                            # write contents
                            print(f"[[[ IN USE##{tidx} ]]] {barrier_title_width} barrier_title_width, {barrier_title_height} barrier_title_height ")
                            # loop through contents within these boundaries and insert them to the canvas
                            for content_idx, content_contour in enumerate(contents_contours):
                                [x, y, w, h] = cv2.boundingRect(content_contour)
                                if x > left_boundary and x < avgwidth and (y+h) < barrier_title_height and y > ct_height:
                                    print(f"WE PRINTED === {content_idx}")
                                    print(f"---> {x} > {left_boundary} and {x} < {avgwidth} and {(y+h)} < {barrier_title_height} and {y} > {ct_height}")
                                    article_mask = cutouts(article_mask, clear_contents_mask, content_contour)
                                    cv2.putText(article_mask, "#{},x{},y{},w{},h{}".format(content_idx, x, y, w, h), cv2.boundingRect(content_contour)[:2], cv2.FONT_HERSHEY_PLAIN, 1.50, [255, 0, 0], 2) #
                                    column_complete = True

                else:
                    # we are here because the title is less than current title of consideration and
                    # therefore ignored :)
                    pass

            if column_complete:
                # a column exit was triggered from the inner loops...
                # choose whether to trigger
                if column_end:
                    # no need to break because we are in the end we should let it continue
                    # incase the article is long and continuing over multiple columns.
                    if avgwidth < ct_width-50:
                        print(f"!! {avgwidth} < {ct_width} !!")
                        multicolumn = True
                    else:
                        multicolumn = False
                        article_end = True

                    col_no += 1
                    left_boundary = avgwidth # store last width
                    avgwidth += x_adjustment
                    print(f"WHAT IS ARTICLE END? {left_boundary} left_boundary {col_no} col_no, {left_boundary} left_boundary, {avgwidth} avgwidth HA!!!!")
                    if multicolumn:
                        continue
                    else:
                        break
                else:
                    # this was triggered procedurally because with hit a title that became a barrier
                    # we therefore don't want to move to next column unnecessarily. Ha!
                    # if current title width is within boundary of this column, then we
                    # should wrap this article up for writing
                    if avgwidth > ct_width:
                        article_end = True
                        # therefore break from going to next columns before wrapping up
                        break
                    # if next title is not within this column then add increments
                    elif nx >= avgwidth and left_boundary < nx:
                        col_no += 1
                        left_boundary = avgwidth # store last width
                        avgwidth += x_adjustment
                    else:
                        article_end = False
            else:
                # article not complete...
                pass

            if article_end:
                break

            # get the boundaries of the current title

        if title_came_up:
            ct_widths.append(cx+cw)
            article_title_p = clear_titles_mask[cy: cy+ch, cx: cx+cw]
            article_mask[cy: cy+ch, cx: cx+cw] = article_title_p # copied title contour onto the blank image

            article_complete = False
        else:
            ct_widths = [] # reset widths
            article_title_p = clear_titles_mask[cy: cy+ch, cx: cx+cw]
            article_mask[cy: cy+ch, cx: cx+cw] = article_title_p # copied title contour onto the blank image

            if (idx+2) == title_count: # we are at the end
                article_title_p = clear_titles_mask[ny: ny+nh, nx: nx+nw]
                article_mask[ny: ny+nh, nx: nx+nw] = article_title_p # copied title contour onto the blank image

            draw_columns(leftmost_x, trimmed_mean, total_columns, article_mask)
            cv2.imwrite(os.path.join(final_directory, 'article_{}.png'.format(idx)), article_mask)
            article_complete = True
            multicolumn = False # reset after a writing

        print("Next start col_no ", col_no)
        if idx == 25:
            sys.exit(0)

    print('Main code {} {}'.format(args.image, args.empty))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cutouts(article_mask, clear_contents_mask, content_contour):
    [x, y, w, h] = cv2.boundingRect(content_contour)
    cv2.drawContours(article_mask, [content_contour], -1, 0, -1)
    cv2.rectangle(article_mask, (x,y), (x+w,y+h), (0, 0, 255), 3)
    contents = clear_contents_mask[y: y+h, x: x+w]
    article_mask[y: y+h, x: x+w] = contents # copied title contour onto the blank image
    return article_mask

def find_column(x_coord, trimmed_mean, leftmost_x, total_columns):
    avgwidth = trimmed_mean + leftmost_x # offset with beginning of first title
    x_adjustment = trimmed_mean

    col_pos = None
    col_no = 1
    # iterate all titles why they are not longer than the current title
    while col_no <= total_columns:
        if x_coord <= avgwidth:
            col_pos = col_no
            break

        col_no += 1
        avgwidth += x_adjustment

    return col_pos

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(prog="LettersIterate", description='Split Veritable Columns in a Newspaper Like Image')
    parser.add_argument('image', type=str, help='Path to the image file') # Required positional argument
    parser.add_argument('--empty', type=bool, help='An optional boolean argument to empty output folder before each processing', default=True) # Optional argument
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    args = parser.parse_args()

    # execute only if run as the entry point into the program
    main(args)