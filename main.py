import random
from pathlib import Path
import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import ImageEnhance
from PIL import Image
import pandas as pd

WIDTH, HEIGHT = 128, 72
TRANSFORM_MAGNITUDE = 10
INPUT_ROOT_DIR = "E:/Datasets/EUVP_Gray"
OUTPUT_ROOT_DIR = "E:/Datasets/EUVP_ChArUco"
#   Colormap for making black and white more grey-ish
gray_colormap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['#2d2d2d', '#d6d6d6'])
#   Black solid background
blackground = Image.new('RGBA', (WIDTH, HEIGHT), (255, 255, 255))
#   Dictionary for 7 x 5 ChArUco
aruco_dict = aruco.custom_dictionary(35, 8)


def find_coeffs(pa, pb):
    """
    https://stackoverflow.com/a/14178717/7773477
    :param pa:  contains four vertices in the resulting plane
    :param pb:  the four vertices in the current plane
    :return:    numpy array of coefficients
    """
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def displace(back_width, back_height, fore_width, fore_height):
    """
    Return  x and y coordinates to place foreground image on background.
    :param back_width:
    :param back_height:
    :param fore_width:
    :param fore_height:
    :return:
    """
    x_var, y_var = round(fore_width / 2), round(fore_height / 2)
    x_base, y_base = round(back_width / 2) - x_var, round(back_height / 2) - y_var
    x_coord = random.randint(x_base - x_var, x_base + x_var)
    y_coord = random.randint(y_base - y_var, y_base + y_var)
    return x_coord, y_coord


def skew(width, height, magnitude, mode='random'):
    """

    :param width:
    :param height:
    :param magnitude:
    :param mode: 0: top narrow, 1: bottom narrow, 2: left skew, 3 right skew
    :return:
    """
    #   Randomize skew
    if mode == 'random':
        mode = random.randint(0, 3)
    #   Translate skew mode into transform coefficients
    if mode == 0:
        coeffs = find_coeffs(
            [(magnitude, 0), (width - magnitude, 0), (width, height), (0, height)],
            [(0, 0), (width, 0), (width, height), (0, height)])
    elif mode == 1:
        coeffs = find_coeffs(
            [(0, 0), (width, 0), (width - magnitude, height), (magnitude, height)],
            [(0, 0), (width, 0), (width, height), (0, height)])
    elif mode == 2:
        coeffs = find_coeffs(
            [(0, 0), (width, 0), (width + magnitude, height), (magnitude, height)],
            [(0, 0), (width, 0), (width, height), (0, height)])
    elif mode == 3:
        coeffs = find_coeffs(
            [(magnitude, 0), (width + magnitude, 0), (width, height), (0, height)],
            [(0, 0), (width, 0), (width, height), (0, height)])
    return coeffs


def rotate():
    """
    Rotate 5, 10, 15, 20, 25 or 30 degrees left or right.
    :return:
    """
    magnitude = random.randint(1, 6) * 5
    if random.randint(0, 1) == 1:
        angle = magnitude
    else:
        angle = 360 - magnitude
    return angle


def charucofy(path_array, source, destination, name, blackground=blackground):
    """
    Place semi-random ChArUco boards on top of the input dataset.
    :param path_array:
    :param source:
    :param destination:
    :param name:
    :param blackground:
    :return:
    """
    #   Create ChArUco from predefined dictionary
    board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)
    #   Draw the board to an array
    imboard = board.draw((WIDTH, HEIGHT))
    #   PIL Image from the array
    pil_img = Image.fromarray(imboard).convert('RGBA')
    #   Solid background
    alpha_composite = Image.alpha_composite(blackground, pil_img)
    #   alpha_composite.save('board_img.png')
    #   Color black and white board more gray-ish
    array_to_color = np.asarray(alpha_composite)
    colored_array = gray_colormap(array_to_color[:, :, 0])
    colored_img = Image.fromarray(np.uint8(colored_array * 255))
    #   colored_img.save('colored_board_img.png')
    #   Transorm / Skew the board
    coefficients = skew(WIDTH, HEIGHT, TRANSFORM_MAGNITUDE)
    transformed = colored_img.transform((WIDTH + TRANSFORM_MAGNITUDE, HEIGHT), Image.PERSPECTIVE, coefficients,
                                        Image.BICUBIC)
    #   transformed.save('transformed_img.png')
    #   Rotate the board, expand image size
    transformed = transformed.rotate(rotate(), expand=1)
    #   transformed.save('rotated_img.png')
    #   Put board on top of background (input) image
    background = Image.open(Path(source, name))
    coordinates = displace(background.size[0], background.size[1], transformed.size[0], transformed.size[1])
    background.paste(transformed, coordinates, transformed.convert('RGBA'))
    #   background.save('dataset_image.png')
    #   Save synthesized image
    current_path = Path(*path_array)
    Path(destination, current_path).mkdir(parents=True, exist_ok=True)
    background.save(Path(destination, current_path, name))
    return "{}, {} OK".format(source, name)


for root, dirs, files in os.walk("{}".format(INPUT_ROOT_DIR)):
    path = root.split(os.sep)
    #   print((len(path) - 1) * '---', os.path.basename(root))
    #   print(path)
    #   print(root)
    #   print(path[1:])
    #   print(Path(*path[1:]))
    #   print(Path(OUTPUT_ROOT_DIR, *path[1:]))
    for file in files:
        # Split the extension from the path and normalise it to lowercase.
        if os.path.splitext(file)[-1].lower() == ".jpg":
            #   print(len(path) * '---', file)
            output = charucofy(path[1:], root, OUTPUT_ROOT_DIR, file)
            print(output)
            #   print(file)
            #   print(os.path.join(root, file))
            #   print(os.path.join(copy_folder, root, file))
