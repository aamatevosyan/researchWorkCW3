import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import convert_blank_image_to_bw, convert_bw_image_to_zero_and_ones, \
    display_image_in_actual_size


def get_contour_stat(contour, image):
    """
    Find mean pixel intensity inside contour

    PARAMETERS
    ----------
    contour : list of x,y coordinates approximating speech box/bubble contour
    image : numpy.ndarray with shape (l, w, c)

    RETURN
    ------
    mean : float

    """
    mask = np.zeros(image.shape, dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean, stddev = cv2.meanStdDev(image, mask=mask)
    return mean


def preprocess_image(in_image, visualize=False):
    """
    Preprocess image for the purpose of text bubble detection

    PARAMETERS
    ----------
    in_image : numpy.ndarray with shape (l, w, c)
    visualize : display original and preprocessed image

    RETURN
    ------
    out_image : numpy.ndarray with shape (l, w, c)

    """

    out_image = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
    out_image = cv2.adaptiveThreshold(out_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 10)
    out_image = cv2.erode(out_image, np.ones((3, 3)), iterations=1)

    if visualize:
        fig, axes = plt.subplots(ncols=2, figsize=(15, 10))
        axes[0].imshow(in_image)
        axes[0].set_title('Original image')
        axes[1].imshow(out_image, cmap='gray')
        axes[1].set_title('Preprocessed image')

    return out_image


def connected_components(image, iom_threshold=0.7, offset=10, visualize=False):
    """
    Rough estimation of bounding boxes using connect_components,
    contains noisy/redundent information, however it always includes
    all speech bubbles / text boxes among estimates so we use this
    as a first pass filter

    PARAMETERS
    ----------
    image : numpy.ndarray with shape (l, w, c)
    iom_threshold : float, determines % of overlap necessary to discard overlapping bounding boxes
    offset : int, extends bounding boxes to include edges of panels

    RETURN
    ------
    box_candidates : list of tuples (y0, y1, x0, x1) denoting bounding box upper-left and bottom-right corners

    """
    box_stats = cv2.connectedComponentsWithStats(image, 4, cv2.CV_32S)[2]
    box_area = box_stats[:, 4]

    l, w = image.shape
    area_condition = (box_area > l * w / 20 ** 2) & (box_area < l * w / 7 ** 2)
    filtered_stats = box_stats[area_condition]

    box_candidates = []
    for x, y, w, h in filtered_stats[:, :4]:
        box_candidates.append((y - offset, y + h + offset, x - offset, x + w + offset))

    if visualize:
        # Messy grid visualization of box_candidates
        nrows, ncols = 5, int(len(box_candidates) / 5) + 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))

        ix = 0
        for row in axes:
            for col in row:
                if ix < len(box_candidates):
                    pts = box_candidates[ix]
                    y0, y1, x0, x1 = pts
                    col.imshow(image[y0:y1, x0:x1], cmap='gray')
                col.get_xaxis().set_ticks([])
                col.get_yaxis().set_ticks([])
                ix += 1
        plt.show()

    return box_candidates


def bubble_contours(image, box_candidates, iom_threshold=0.5, convexify=False, visualize=False):
    """
    Find contours of bounding boxes in box_candidates, refine boundary estimates & discard redundency

    PARAMETERS
    ----------
    image : numpy.ndarray with shape (l, w, c)
    box_candidates : list of tuples, (y0,y1,x0,x1) denoting bounding box upper-left and bottom-right corners
    iom_threshold : float, determines % of overlap necessary to discard overlapping bounding boxes
    convexify : bool, indicates if contour estimates of speech bubbles should be convex or not
    visualize : bool, display segmentation map and preprocessed image

    RETURN
    ------
    box_stats : list of 2-tuples (contour_coordinates, bounding_box_coordinates)

    """

    draw_mask = np.zeros_like(image)

    box_stats = []
    contours = []

    for y0, y1, x0, x1 in box_candidates:
        # Find contours of speech bubbles in connected components (rectangular boxes)
        mask = np.zeros_like(image)
        mask[y0:y1, x0:x1] = image[y0:y1, x0:x1]
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in cnts:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

            # Check pixel intensity and discard those lying outside normal (heuristic) range
            if get_contour_stat(cnt, image) > 240 or get_contour_stat(cnt, image) < 100:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cnt_area = cv2.contourArea(cnt)
            circle_radius = cv2.minEnclosingCircle(cnt)[1]
            circle_area_ratio = int(3.14 * circle_radius ** 2 / (cnt_area + 1e-6))
            rect_area_ratio = int(w * h / cnt_area)
            # This is a speech "bubble" heuristic, it should also work for boxes
            # The basic idea is that a bubbles area should approximate that of an enclosing circle
            if ((circle_area_ratio <= 4) & (cnt_area > 4000)) or (rect_area_ratio == 1):
                if convexify:
                    approx = cv2.convexHull(approx)
                contours.append(cnt)
                box_stats.append((approx, (y, y + h, x, x + w)))
                cv2.fillPoly(draw_mask, [approx], (255, 255, 255))

    #    # Remove overlapping boxes
    #    coordinates = [pts for _, pts in box_stats]
    #    coordinate_pairs = itertools.combinations(coordinates, 2)

    #    for bb1, bb2 in coordinate_pairs:
    #        dict1 = dict(zip(['y1', 'y2', 'x1', 'x2'], bb1))
    #        dict2 = dict(zip(['y1', 'y2', 'x1', 'x2'], bb2))

    #        iom, bigger_ix = calculate_iom(dict1, dict2)
    #        if iom > iom_threshold:
    #            if bigger_ix == 0:
    #                smaller = bb2
    #            else:
    #                smaller = bb1
    #            if smaller in coordinates:
    #                smaller_ix = coordinates.index(smaller)
    #                del coordinates[smaller_ix]
    #                del box_stats[smaller_ix]

    if visualize:
        fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
        ax[0].imshow(draw_mask, cmap='gray')
        ax[1].imshow(image, cmap='gray')
        plt.tight_layout()

    # return box_stats
    return (draw_mask, box_stats)


def opencv_advanced_method(img_path: str):
    npimg = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    height, width, _ = img.shape

    img = preprocess_image(img)
    box_candidates = connected_components(img, visualize=False)
    blank_image, _ = bubble_contours(img, box_candidates, convexify=False, visualize=False)
    bw = convert_blank_image_to_bw(blank_image)

    return convert_bw_image_to_zero_and_ones(bw)
