#!/usr/bin/env python
import os
import sys
import glob
import logging
from pathlib import Path
from pprint import pprint
from datetime import datetime
import math
import argparse
import cv2
import imutils
import numpy as np
from scipy import stats
from pythonRLSA import rlsa
import pytesseract
from PIL import Image
import xml.etree.cElementTree as ET
from xml.dom import minidom

from utils import lines_extraction, draw_lines, extract_polygons, \
    column_summaries, determine_precedence, redraw_titles, redraw_contents, \
    draw_columns, cutouts

def process_image(path_to_image, empty_output, output_dir):
    output_path = os.path.dirname(path_to_image)
    last_folder_name = os.path.basename(output_path)
    image_name = os.path.basename(path_to_image)
    image_sans_ext = os.path.splitext(image_name)[0]

    # check if file exists here and exist if not
    try:
        f = open(path_to_image)
        f.close()
    except FileNotFoundError:
        logging.critical('Given image does not exist')
        sys.exit(0)

    logging.info(f"Processing {image_name}")

    founds = glob.glob(f'{output_dir}/{image_sans_ext}-*.xml')
    if len(founds) > 0:
        logging.info(f"FILE EXISTS: {founds}")
        return

    # standardize size of the images maintaining aspect ratio
    if empty_output:
        files = glob.glob('{}/*'.format(output_dir))
        for f in files:
            os.remove(f)

    image = cv2.imread(path_to_image) #reading the image

    image_height = image.shape[0]
    image_width = image.shape[1]
    if image_width != 2048:
        image = imutils.resize(image, width=2048)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting to grayscale image
    # applying thresholding technique on the grayscale image
    # all pixels value above 0 will be set to 255 but because we are using THRESH_OTSU
    # we have avoid have to set threshold (i.e. 0 = just a placeholder) since otsu's method does it automatically
    (thresh, im_bw) = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # converting to binary image
    # invert image data using unary tilde operator
    # im_bw = ~im_bw

    # Noise removal step - Perform opening on the thresholded image (erosion followed by dilation)
    kernel = np.ones((2,2),np.uint8) # kernel noise size (2,2)
    im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel) # cleans up random lines that appear on the page
    if logging.getLogger().level == logging.DEBUG: cv2.imwrite(os.path.join(output_dir, f'{image_sans_ext}-im-negative.png'), im_bw)
    if logging.getLogger().level == logging.DEBUG: cv2.imwrite(os.path.join(output_dir, f'{image_sans_ext}-im-bw.png'), ~im_bw)

    # extract and draw any lines from the image
    lines_mask = draw_lines(image, gray)
    if logging.getLogger().level == logging.DEBUG: cv2.imwrite(os.path.join(output_dir, f'{image_sans_ext}-lines-mask.png'), lines_mask) # debug remove

    # extract complete shapes likes boxes of ads and banners
    found_polygons_mask = extract_polygons(im_bw, lines_mask)
    if logging.getLogger().level == logging.DEBUG: cv2.imwrite(os.path.join(output_dir, f'{image_sans_ext}-found-polygons-mask.png'), found_polygons_mask) # debug remove

    # nullifying the mask of unwanted polygons over binary (toss images)
    # this should not only have texts, without images
    text_im_bw = cv2.bitwise_and(im_bw, im_bw, mask=found_polygons_mask)
    if logging.getLogger().level == logging.DEBUG: cv2.imwrite(os.path.join(output_dir, f'{image_sans_ext}-text-im-bw-negative.png'), ~text_im_bw)

    # initialize blank image for extracted titles
    titles_mask = np.ones(image.shape[:2], dtype="uint8") * 255
    contents_mask = np.ones(image.shape[:2], dtype="uint8") * 255

    (contours, _) = cv2.findContours(text_im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(contour)[3] for contour in contours]
    avgheight = sum(heights)/len(heights)

    title_widths = []
    content_widths = []
    if logging.getLogger().level == logging.DEBUG: debug_contents_mask = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image for debug colour
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
        if logging.getLogger().level == logging.DEBUG:
            cv2.drawContours(debug_contents_mask, [c], -1, 0, -1)
            cv2.rectangle(debug_contents_mask, (x,y), (x+w,y+h), (0, 255, 0), 1)
    if logging.getLogger().level == logging.DEBUG: cv2.imwrite(os.path.join(output_dir, f'{image_sans_ext}-debug_drawn_contours.png'), debug_contents_mask)

    # helps further detach titles if necessary. This step can be removed
    # titles_mask = cv2.erode(titles_mask, kernel, iterations = 1)
    m_height, m_width = titles_mask.shape # get image dimensions, height and width

    # make 2D Image mask of proto-original image for cutting contents
    image_mask = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image
    image_mask[0: m_height, 0: m_width] = image[0: m_height, 0: m_width]

    # run length smoothing algorithms for vertical and lateral conjoining of pixels
    value = math.ceil(sum(title_widths)/len(title_widths))*2
    logging.info(f'RLSA Title Value {value}')
    rlsa_titles_mask = rlsa.rlsa(titles_mask, True, False, value) #rlsa application
    rlsa_titles_mask_for_final = rlsa_titles_mask
    if logging.getLogger().level == logging.DEBUG: cv2.imwrite(os.path.join(output_dir, f'{image_sans_ext}-rlsa-titles-mask.png'), rlsa_titles_mask) # debug remove

    value = math.ceil(sum(content_widths)/len(content_widths))*3
    logging.info(f'RLSA Content Value {value}')
    rlsa_contents_mask = rlsa.rlsa(contents_mask, False, True, value) #rlsa application
    rlsa_contents_mask_for_avg_width = rlsa_contents_mask
    if logging.getLogger().level == logging.DEBUG: cv2.imwrite(os.path.join(output_dir, f'{image_sans_ext}-rlsa-contents-mask.png'), rlsa_contents_mask) # debug remove

    # get avg properties of columns
    contents_sum_list, contents_x_list, for_avgs_contours_mask = column_summaries(image, rlsa_contents_mask_for_avg_width)
    if logging.getLogger().level == logging.DEBUG: cv2.imwrite(os.path.join(output_dir, f'{image_sans_ext}-for-avgs-contours-mask.png'), for_avgs_contours_mask) # debug remove
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

    # draw_columns(leftmost_x, trimmed_mean, total_columns, clear_titles_mask)
    if logging.getLogger().level == logging.DEBUG: cv2.imwrite(os.path.join(output_dir, f'{image_sans_ext}-clear-titles-mask.png'), clear_titles_mask) # debug remove

    ### contents work
    (contours, _) = cv2.findContours(~rlsa_contents_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # apply some heuristic to different other stranger things masquerading as titles
    nt_contours = [contour for contour in contours if cv2.boundingRect(contour)[2]*cv2.boundingRect(contour)[3] > threshold]

    contents_contours = sorted(nt_contours, key=lambda contour:determine_precedence(contour, total_columns, trimmed_mean, leftmost_x, m_height))
    clear_contents_mask = redraw_contents(image_mask, contents_contours)
    # draw_columns(leftmost_x, trimmed_mean, total_columns, clear_contents_mask)
    if logging.getLogger().level == logging.DEBUG: cv2.imwrite(os.path.join(output_dir, f'{image_sans_ext}-sorted-clear-contents-mask.png'), clear_contents_mask)

    # start printing individual letters based on titles! The final act
    (contours, _) = cv2.findContours(~rlsa_titles_mask_for_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # apply some heuristic to different other stranger things masquerading as titles
    nt_contours = [contour for contour in contours if cv2.boundingRect(contour)[2]*cv2.boundingRect(contour)[3] > threshold]

    contours = sorted(nt_contours, key=lambda contour:determine_precedence(contour, total_columns, trimmed_mean, leftmost_x, m_height))

    article_complete = False
    title_came_up = True
    title_count = len(contours)
    ct_widths = []
    article_mask = np.ones(image.shape, dtype="uint8") * 255 # blank layer image for one article
    letter_root = ET.Element("letter")

    desc = ET.SubElement(letter_root, "description")
    ET.SubElement(desc, "MeasurementUnit").text = "pixel"
    ocv_proc = ET.SubElement(desc, "OPenCVProcessing", pageImage=image_sans_ext)
    ET.SubElement(ocv_proc, "ProcessingDateTime").text = str(datetime.today())
    ET.SubElement(ocv_proc, "Script").text = 'Lettersiterate'

    layout = ET.SubElement(letter_root, "Layout")
    page = ET.SubElement(layout, "Page")
    print_space = ET.SubElement(page, "PrintSpace", height=str(image_height), width=str(image_width), xpos=str(0), ypos=str(0))
    # ET.Element(print_space, attrib={'height':image_height, 'width':image_width, 'xpos':0, 'ypos':0})

    # for idx, contour in enumerate(contours):
    for idx, (_curr, _next) in enumerate(zip(contours[::],contours[1::])):
        # https://www.quora.com/How-do-I-iterate-through-a-list-in-python-while-comparing-the-values-at-adjacent-indices/answer/Jignasha-Patel-14
        if article_complete:
            article_mask = np.ones(image.shape, dtype="uint8") * 255 # blank layer image for another separate letter

            # xml file
            letter_root = ET.Element("letter")

            desc = ET.SubElement(letter_root, "description")
            ET.SubElement(desc, "MeasurementUnit").text = "pixel"
            ocv_proc = ET.SubElement(desc, "OPenCVProcessing")
            ET.SubElement(ocv_proc, "ProcessingDateTime").text = str(datetime.today())
            ET.SubElement(ocv_proc, "Script").text = 'Lettersiterate'

            layout = ET.SubElement(letter_root, "Layout")
            page = ET.SubElement(layout, "Page")
            print_space = ET.SubElement(page, "PrintSpace", height=str(image_height), width=str(image_width), xpos=str(0), ypos=str(0))

        [cx, cy, cw, ch] = cv2.boundingRect(_curr)
        [nx, ny, nw, nh] = cv2.boundingRect(_next)

        ct_height = cy+ch # title height in this column
        ct_widths.append(cx+cw)
        ct_width = max(ct_widths) # adjust to get longest title width if multiple line title :)

        # dont proceed any further if the next title is right below it on same column
        # continue to next title
        # current and next have to be within the same column
        # detect last article in the columns
        if (idx+2) == title_count:
            title_came_up = False
        elif cy < ny and ny-(nh*3) < cy and nx < ct_width:
            # 1) current title is above next
            # 2) next title is directly above current
            # 3) next title is withing the length of the current title. Cannot be in another column
            # and considered directly below current. Phew!, it happened
            title_came_up = True
        else:
            title_came_up = False

        if not title_came_up:
            title_encounters = 0
            # loop through contents within these boundaries and insert them to the canvas
            for content_idx, content_contour in enumerate(contents_contours):
                [x, y, w, h] = cv2.boundingRect(content_contour)
                content_width = x+w
                # length -50 is to be safe sometimes the content cut maybe infringe onto the next title
                # get any content that starts within the title (take out -50) and within the end of the title width
                # and give (+50), it is still below the title
                logging.debug(f"{x} >= {cx-50} and {x} <= {ct_width} and {y+50} > {ct_height}")
                if x >= cx-50 and x <= ct_width and y+50 > ct_height:
                    # now that we have limited the content to be within the width and below the title of interest
                    # make sure it does not transgress into other titles. The bane of my existence begins, sigh!
                    for tidx, tcontour in enumerate(contours):
                        [tx, ty, tw, th] = cv2.boundingRect(tcontour)
                        # validating titles that are directly below
                        # 1) it has to be greater than the current title
                        # 2) it starts within the width of the current title
                        # 3) it starts within the width of the current content
                        # 4) it does not start left of the content even if we take out 50 pixels to the left (-50)
                        if tidx > idx and tx < ct_width and tx < content_width and tx > x-50 and title_encounters < 1:
                            # print(f"TITLE BELOW---> ###{content_idx} ##{tidx} > #{idx} and {tx} < {content_width} and {cx} >= {x-50}")
                            article_mask = cutouts(article_mask, clear_contents_mask, content_contour)
                            ET.SubElement(print_space, "BodyText", height=str(h), width=str(w), xpos=str(x), ypos=str(y), contourId=str(idx), bodyTextContourId=str(content_idx))
                            # cv2.putText(article_mask, "###{content_idx},{x},{y}.{w},{h}", cv2.boundingRect(content_contour)[:2], cv2.FONT_HERSHEY_PLAIN, 1.50, [255, 0, 0], 2)
                            title_encounters += 1
                            # hitting a title in this case means we don't need to go any further for current content
                            break

                        # validating titles that are on a different column
                        # 1)it has to be greater than the current title
                        # 2)it starts within the width of the current title
                        # 3)it starts below this content but within the contents limits (meaning it is multicolumn extension)
                        if tidx > idx and tx < ct_width and (ty > y and tx > x-50) and title_encounters < 1:
                            article_mask = cutouts(article_mask, clear_contents_mask, content_contour)
                            ET.SubElement(print_space, "BodyText", height=str(h), width=str(w), xpos=str(x), ypos=str(y), contourId=str(idx), bodyTextContourId=str(content_idx))

                    # validating titles that are at the end of the column
                    # 1) there is no title directly below it
                    if all(x < cv2.boundingRect(tcontour)[0] for tidx, tcontour in enumerate(contours) if tidx > idx and cv2.boundingRect(tcontour)[0] > content_width) and title_encounters < 1:
                        article_mask = cutouts(article_mask, clear_contents_mask, content_contour)
                        ET.SubElement(print_space, "BodyText", height=str(h), width=str(w), xpos=str(x), ypos=str(y), contourId=str(idx), bodyTextContourId=str(content_idx))

        if title_came_up:
            ct_widths.append(cx+cw)
            article_title_p = clear_titles_mask[cy: cy+ch, cx: cx+cw]
            article_mask[cy: cy+ch, cx: cx+cw] = article_title_p # copied title contour onto the blank image
            ET.SubElement(print_space, "Title", height=str(ch), width=str(cw), xpos=str(cx), ypos=str(cy), contourId=str(idx))
            article_complete = False
        else:
            ct_widths = [] # reset widths
            article_title_p = clear_titles_mask[cy: cy+ch, cx: cx+cw]
            article_mask[cy: cy+ch, cx: cx+cw] = article_title_p # copied title contour onto the blank image
            ET.SubElement(print_space, "Title", height=str(ch), width=str(cw), xpos=str(cx), ypos=str(cy), contourId=str(idx))

            if (idx+2) == title_count: # we are at the end
                article_title_p = clear_titles_mask[ny: ny+nh, nx: nx+nw]
                article_mask[ny: ny+nh, nx: nx+nw] = article_title_p # copied title contour onto the blank image

            file_name = f"article-{str(idx).zfill(2)}"
            if logging.getLogger().level == logging.DEBUG: cv2.imwrite(os.path.join(output_dir, f"{image_sans_ext}-{file_name}.png"), article_mask)
            article_complete = True

            content = pytesseract.image_to_string(Image.fromarray(article_mask))
            with open(os.path.join(output_dir, f'{image_sans_ext}-{file_name}.txt'), 'a') as the_file:
                the_file.write(content)
            ET.SubElement(page, "TextBlock", articleNo=str(file_name), contourId=str(idx)).text = content

            tree = ET.ElementTree(letter_root)
            xml_output_file = os.path.join(output_dir, f'{image_sans_ext}-{file_name}.xml')
            # this method may cause 'OSError: [Errno 24] Too many open files' and does not prettyprint
            # tree.write(xml_output_file, encoding='utf8')
            # OR 
            xmlstr = ET.tostring(letter_root).decode()
            xmlstr = minidom.parseString(xmlstr).toprettyxml(indent="\t", newl="\n")
            with open(xml_output_file,'w+') as outfile:
                outfile.write(xmlstr)         

def main(args):
    # get params
    path_to_image = args.image
    path_to_dir = args.dir
    empty_output = args.empty
    
    # create out dir
    current_directory = os.getcwd()
    output_path = os.path.join(current_directory, r'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)    

    if path_to_dir:
        done_list = [str(f) for f in sorted(Path(output_path).glob('*.xml'))]
        # initialize check until none is found since it is sequentially done
        keep_checking = True
        
        for f in sorted(Path(path_to_dir).glob('**/*.png')):
            f = str(f) # cast PosixPath to str
            if "page-8" in f:
                txt_name = os.path.basename(f)
                txt_sans_ext = os.path.splitext(txt_name)[0]
                if any(txt_sans_ext in file_path for file_path in done_list):
                    logging.info(f"FILE STARTING: {txt_sans_ext}... PROCESSED")
                    continue
                else:
                    # means we have alread found starting point
                    keep_checking = False
                    
           
                process_image(f, empty_output, output_path)
    elif path_to_image:
        process_image(path_to_image, empty_output, output_path)

    print('Main code {} {}'.format(args.image, args.empty))

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(prog="LettersIterate", description='Split Veritable Columns in a Newspaper Like Image')
    parser.add_argument('--image', type=str, help='Path to the image file')
    parser.add_argument('--dir', type=str, help='An optional path to directory with newspaper images') # Optional argument
    parser.add_argument('--empty', dest='empty', action='store_true', help='An optional boolean argument to empty output folder before each processing')
    parser.add_argument('--no-empty', dest='empty', action='store_false', help='An optional boolean argument to NOT empty output folder before each processing')
    parser.add_argument('--debug', dest="loglevel", action='store_true', help='print debug messages to stderr')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.print_help(sys.stderr)
        sys.exit(1)


    if args.loglevel:
        logging.basicConfig(level=logging.DEBUG) # 10
    else:
        logging.basicConfig(level=logging.INFO) # 20

    # execute only if run as the entry point into the program
    main(args)
