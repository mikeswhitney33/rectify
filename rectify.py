"""
blackboard.py

this takes in an image of a blackboard and rectifies it
"""
import argparse

import numpy as np
import cv2 as cv

LONG_SIDE = 15
SHORT_SIDE = 3

def draw_topleft(image, point, color):
    """
    draws a marking for a top left rectangle
    """
    x_coord, y_coord = point
    cv.rectangle(
        image,
        (x_coord-SHORT_SIDE, y_coord-SHORT_SIDE),
        (x_coord+LONG_SIDE, y_coord),
        color,
        -1)
    cv.rectangle(
        image,
        (x_coord-SHORT_SIDE, y_coord-SHORT_SIDE),
        (x_coord, y_coord+LONG_SIDE),
        color,
        -1)

def draw_topright(image, point, color):
    """
    draws a marking for a top right rectangle
    """
    x_coord, y_coord = point
    cv.rectangle(
        image,
        (x_coord+SHORT_SIDE, y_coord-SHORT_SIDE),
        (x_coord-LONG_SIDE, y_coord),
        color,
        -1)
    cv.rectangle(
        image,
        (x_coord+SHORT_SIDE, y_coord-SHORT_SIDE),
        (x_coord, y_coord+LONG_SIDE),
        color,
        -1)

def draw_bottomleft(image, point, color):
    """
    draws a marking for a bottom left rectangle
    """
    x_coord, y_coord = point
    cv.rectangle(
        image,
        (x_coord-SHORT_SIDE, y_coord+SHORT_SIDE),
        (x_coord+LONG_SIDE, y_coord),
        color,
        -1)
    cv.rectangle(
        image,
        (x_coord-SHORT_SIDE, y_coord+SHORT_SIDE),
        (x_coord, y_coord-LONG_SIDE),
        color,
        -1)

def draw_bottomright(image, point, color):
    """
    draws a marking for a bottom right rectangle
    """
    x_coord, y_coord = point
    cv.rectangle(
        image,
        (x_coord+SHORT_SIDE, y_coord+SHORT_SIDE),
        (x_coord-LONG_SIDE, y_coord),
        color,
        -1)
    cv.rectangle(
        image,
        (x_coord+SHORT_SIDE, y_coord+SHORT_SIDE),
        (x_coord, y_coord-LONG_SIDE),
        color,
        -1)


def resize_aspect(image, width=None, height=None):
    """ resizes an image but keeps aspect ratio """
    if width is None and height is None:
        return image
    im_height, im_width = image.shape[:2]
    if width is None:
        ratio = height / im_height
        dim = (ratio * im_height, height)
    elif height is None:
        ratio = width / im_width
        dim = (width, ratio * im_width)
    else:
        dim = (width, height)
    dim = tuple(map(round, dim))
    return cv.resize(image, dim)


def get_rough_size(points):
    """ gets the rough width and height of the points selected """
    minx, miny, maxx, maxy = 1e8, 1e8, -1e8, -1e8
    for (x_coord, y_coord) in points:
        minx = min(x_coord, minx)
        miny = min(y_coord, miny)
        maxx = max(x_coord, maxx)
        maxy = max(y_coord, maxy)
    return (round(maxx-minx), round(maxy-miny))

DRAWING_FUNCS = [draw_topleft, draw_topright, draw_bottomleft, draw_bottomright]

def draw_points(image, points, current):
    """ draws the points """
    for i, point in enumerate(points):
        color = (0, 72, 0)
        if i == current:
            color = (0, 255, 255)
        DRAWING_FUNCS[i](image, point, color)
    line_color = (0, 72, 0)
    topleft, topright, bottomleft, bottomright = points
    cv.line(image, topleft, topright, line_color, 1)
    cv.line(image, topright, bottomright, line_color, 1)
    cv.line(image, bottomright, bottomleft, line_color, 1)
    cv.line(image, bottomleft, topleft, line_color, 1)

def select_point(input_point, points, thresh=25):
    """ selects the nearset point """
    input_x, input_y = input_point
    for i, point in enumerate(points):
        x_coord, y_coord = point
        dist = np.sqrt(np.square(x_coord-input_x) + np.square(y_coord-input_y))
        if dist < thresh:
            return i
    return -1

def select_points(input_image):
    """
    allows the users to select 4 points from the image
    """
    image = resize_aspect(input_image, width=600)
    print("Select the 4 points to rectify. press enter when done")
    points = [
        (SHORT_SIDE, SHORT_SIDE),
        (image.shape[1]-SHORT_SIDE, SHORT_SIDE),
        (SHORT_SIDE, image.shape[0]-SHORT_SIDE),
        (image.shape[1]-SHORT_SIDE, image.shape[0]-SHORT_SIDE)
    ]
    current = -1
    tmp = image.copy()
    def mouse_callback(_event, x_coord, y_coord, flags, _userdata):
        """ local function for mouse callback """
        nonlocal current
        if flags == cv.EVENT_FLAG_LBUTTON:
            current = select_point((x_coord, y_coord), points)
            if current >= 0:
                points[current] = (x_coord, y_coord)

    cv.namedWindow("points")
    cv.setMouseCallback("points", mouse_callback)
    while True:
        tmp = image.copy()
        draw_points(tmp, points, current)
        cv.imshow("points", tmp)
        key = cv.waitKey(5)
        if key & 0xFF == ord('s'):
            break

    og_size = input_image.shape
    x_ratio = og_size[1] / image.shape[1]
    y_ratio = og_size[0] / image.shape[0]

    points = [(x * x_ratio, y * y_ratio) for x, y in points]
    return points

def rectify(imname, output):
    """ rectifies the image """
    og_image = cv.imread(imname, cv.IMREAD_COLOR)
    points = select_points(og_image)
    size = get_rough_size(points)
    h_points = [(0, 0), (size[0], 0), (0, size[1]), (size[0], size[1])]
    homography, _ = cv.findHomography(np.array(points), np.array(h_points))
    result = cv.warpPerspective(og_image, homography, size)
    cv.namedWindow("result", cv.WINDOW_KEEPRATIO)
    while True:
        cv.imshow("result", result)
        key = cv.waitKey(5)
        if key & 0xFF == ord('s'):
            if output is None:
                output = f"rectified_{imname}"
            cv.imwrite(output, result)
            print(f"Saved {output}")
            break
        if key & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()


def main():
    """ the main program"""
    parser = argparse.ArgumentParser()
    parser.add_argument("imname")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rectify(args.imname, args.output)


if __name__ == "__main__":
    main()
