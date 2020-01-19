#!/usr/bin/env python
import os
import sys
import csv
import glob
import logging as log
import math
import argparse
import cv2
import imutils
import numpy as np
from pythonRLSA import rlsa
import pytesseract
from PIL import Image
from scipy.ndimage import interpolation as inter

from utils import draw_lines, extract_polygons, cutouts


def correct_skew(image, delta=1, limit=5):
    """
    Correct skewing in some cases
    Credit: https://stackoverflow.com/a/58226633/754432
    """
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV +
                           cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated


def process_image(path_to_image, empty_output, out_dir_name):
    image_name = os.path.basename(path_to_image)
    img_sans_ext = os.path.splitext(image_name)[0]

    # check if file exists here and exist if not
    try:
        f = open(path_to_image)
        f.close()
    except FileNotFoundError:
        log.critical('Given image does not exist')
        sys.exit(0)

    log.info(f"Processing {image_name}")

    # create out dir
    current_directory = os.getcwd()
    final_dir = os.path.join(current_directory, r'dates')
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)

    founds = glob.glob(f'{final_dir}/{img_sans_ext}-*.xml')
    if len(founds) > 0:
        log.info(f"FILE EXISTS: {founds}")
        return

    # standardize size of the images maintaining aspect ratio
    if empty_output:
        files = glob.glob('{}/*'.format(final_dir))
        for f in files:
            os.remove(f)

    image = cv2.imread(path_to_image)  # reading the image

    image_width = image.shape[1]
    if image_width != 2048:
        image = imutils.resize(image, width=2048)

    gray = cv2.cvtColor(image,
                        cv2.COLOR_BGR2GRAY)  # converting to grayscale image
    # applying thresholding technique on the grayscale image
    # all pixels value above 0 will be set to 255 but because
    # we are using THRESH_OTSU
    # we have avoid have to set threshold (i.e. 0 = just a placeholder)
    # since otsu's method does it automatically
    (thresh, im_bw) = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # converting to binary image
    # invert image data using unary tilde operator
    # im_bw = ~im_bw

    # Noise removal step - Perform opening on the thresholded image
    # (erosion followed by dilation)
    kernel = np.ones((2, 2), np.uint8)  # kernel noise size (2,2)
    # cleans up random lines that appear on the page
    im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)
    if log.getLogger().level == log.DEBUG:
        cv2.imwrite(os.path.join(final_dir,
                                 f'{img_sans_ext}-im-negative.png'), im_bw)
    if log.getLogger().level == log.DEBUG:
        cv2.imwrite(os.path.join(final_dir,
                                 f'{img_sans_ext}-im-bw.png'), ~im_bw)

    # extract and draw any lines from the image
    lines_mask = draw_lines(image, gray)
    if log.getLogger().level == log.DEBUG:
        cv2.imwrite(os.path.join(final_dir,
                                 f'{img_sans_ext}-lines-mask.png'), lines_mask)

    # extract complete shapes likes boxes of ads and banners
    found_polygons_mask = extract_polygons(im_bw, lines_mask)
    if log.getLogger().level == log.DEBUG:
        cv2.imwrite(
            os.path.join(final_dir, f'{img_sans_ext}-found-polygons-mask.png'),
            found_polygons_mask)

    # nullifying the mask of unwanted polygons over binary (toss images)
    # this should not only have texts, without images
    text_im_bw = cv2.bitwise_and(im_bw, im_bw, mask=found_polygons_mask)
    if log.getLogger().level == log.DEBUG:
        cv2.imwrite(
            os.path.join(final_dir, f'{img_sans_ext}-text-im-bw-negative.png'),
            ~text_im_bw)

    # initialize blank image for extracted contents
    contents_mask = np.ones(image.shape[:2], dtype="uint8") * 255

    (contours, _) = cv2.findContours(text_im_bw,
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(contour)[3] for contour in contours]
    avgheight = sum(heights)/len(heights)

    content_widths = []
    if log.getLogger().level == log.DEBUG:
        # blank 3 layer image for debug colour
        debug_mask = np.ones(image.shape, dtype="uint8") * 255
    # finding the larger text
    for c in contours:
        [x, y, w, h] = cv2.boundingRect(c)
        cv2.rectangle(contents_mask, (x, y), (x+w, y+h), (255, 0, 0), 1)
        if h > 2*avgheight:  # avoid titles altogether
            pass
        elif h*w > 20 and x > 1000 and y < 100:  # avoid specks or dots
            # get the biggest chunks of texts... articles!
            cv2.drawContours(contents_mask, [c], -1, 0, -1)
            content_widths.append(w)

        if log.getLogger().level == log.DEBUG:
            cv2.drawContours(debug_mask, [c], -1, 0, -1)
            cv2.rectangle(debug_mask, (x, y), (x+w, y+h), (0, 255, 0), 1)
    if log.getLogger().level == log.DEBUG:
        cv2.imwrite(os.path.join(
            final_dir, f'{img_sans_ext}-debug_drawn_contours.png'),
            debug_mask)

    # get image dimensions, height and width
    m_height, m_width = contents_mask.shape

    # make 2D Image mask of proto-original image for cutting contents
    # blank 3 layer image
    image_mask = np.ones(image.shape, dtype="uint8") * 255
    image_mask[0: m_height, 0: m_width] = image[0: m_height, 0: m_width]

    try:
        value = math.ceil(sum(content_widths)/len(content_widths))*5
    except ZeroDivisionError as e:
        value = 140
    log.info(f'RLSA Content Value {value}')
    # rlsa application
    rlsa_contents_mask = rlsa.rlsa(contents_mask, True, False, value)
    if log.getLogger().level == log.DEBUG:
        cv2.imwrite(os.path.join(
            final_dir, f'{img_sans_ext}-rlsa-contents-mask.png'),
            rlsa_contents_mask)  # debug remove

    threshold = 1500  # remove tiny contours that dirtify the image

    # contents work
    (contours, _) = cv2.findContours(~rlsa_contents_mask,
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    # apply some heuristic to different other stranger things
    # masquerading as contents
    contents_contours = [contour for contour in contours if
                         cv2.boundingRect(contour)[2] *
                         cv2.boundingRect(contour)[3] > threshold]

    # blank layer image for one article
    article_mask = np.ones(image.shape, dtype="uint8") * 255

    # loop through and insert it to the canvas
    for content_idx, content_contour in enumerate(contents_contours):
        # https://www.quora.com/How-do-I-iterate-through-a-list-in-python-while-comparing-the-values-at-adjacent-indices/answer/Jignasha-Patel-14

        [x, y, w, h] = cv2.boundingRect(content_contour)

        if x > 1000 and y < 100:
            log.debug(f"{x} >= {x-50} and {x} {y+50}")
            article_mask = cutouts(article_mask, image_mask, content_contour)

    angle, rotated_article_mask = correct_skew(article_mask)
    log.info(f'Rotation Angle: {angle}')

    # DIlating the output improved overall readbility by tesseract especially
    # in cases where resulting output was empty
    # https://stackoverflow.com/a/54582118/754432
    cv2.dilate(rotated_article_mask, (5, 5), rotated_article_mask)

    if log.getLogger().level == log.DEBUG:
        cv2.imwrite(os.path.join(final_dir,
                                 f"{img_sans_ext}.png"), rotated_article_mask)

    # 3 Fully automatic page segmentation, but no OSD. (default for tesserocr)
    # 7 means treat the image as a single text line.
    # https://medium.com/better-programming/beginners-guide-to-tesseract-ocr-using-python-10ecbb426c3d
    content = pytesseract.image_to_string(
        Image.fromarray(rotated_article_mask),
        config='--psm 3')

    with open(os.path.join(final_dir, f'{out_dir_name}.csv'), 'a+') as f_out:
        # Using dictionary keys as fieldnames for the CSV file header
        writer = csv.writer(f_out, delimiter='\t')
        # writer = csv.DictWriter(f_out, fieldnames=['file_name', 'raw_date'])
        writer.writerow([img_sans_ext, content.partition('\n')[0]])


def main(args):
    # get params
    path_to_image = args.image
    path_to_dir = args.dir
    empty_output = args.empty

    if path_to_dir:
        out_dir_name = os.path.basename(path_to_dir)
        for f in sorted(glob.glob('{}/**/*.png'.format(path_to_dir),
                        recursive=True)):
            if "page-8" in f:
                process_image(f, empty_output, out_dir_name)
    elif path_to_image:
        output_path = os.path.dirname(path_to_image)
        last_folder_name = os.path.basename(output_path)

        process_image(path_to_image, empty_output, last_folder_name)

    print('Main code {} {}'.format(args.image, args.empty))


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(
        prog="LettersIterate",
        description='Split Veritable Columns in a Newspaper Like Image')
    parser.add_argument('--image', type=str, help='Path to the image file')
    # Optional argument
    parser.add_argument(
        '--dir',
        type=str,
        help='An optional path to directory with newspaper images')
    # Optional argument
    parser.add_argument(
        '--empty',
        dest='empty',
        action='store_true',
        help='Boolean argument to empty output dir before each processing')
    parser.add_argument(
        '--no-empty',
        dest='empty',
        action='store_false',
        help='Boolean argument to NOT empty output dir before each processing')
    parser.add_argument(
        '--debug',
        dest="loglevel",
        action='store_true',
        help='print debug messages to stderr')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.loglevel:
        log.basicConfig(level=log.DEBUG)  # 10
    else:
        log.basicConfig(level=log.INFO)  # 20

    # execute only if run as the entry point into the program
    main(args)
