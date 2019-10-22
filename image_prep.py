# import required packages
import face_recognition
import cv2
import imutils
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Optional, Tuple, Union
import json


def json_writer(data, json_file):
    """writes to json file"""
    try:
        with open(json_file, mode='w') as open_file:
            json.dump(data, open_file)
            open_file.write('\n')
    except Exception as e:
        print(e)


def load_rgb_image(raw_image):
    """
    Function to load image and then convert it into rgb (opencv has bgr order)
    :param raw_image: raw image data in bytes format
    :return: rgb image
    """
    min_res = 20
    max_res = 11500
    # read the image
    if raw_image is not None:
        # Convert rawImage to Mat
        pil_image = Image.open(BytesIO(raw_image))
        np_image = np.array(pil_image)
        ###

        ### To accommodate with the low resolution we are getting from the camera
        width = 215
        height = 215
        dim = (width, height)
        np_image = cv2.resize(np_image, dim, interpolation=cv2.INTER_AREA)
        ###

        # check if it is in proper image format (i.e it is of shape 2(b&w) or 3(gray or colour image))
        if len(np_image.shape) in [2, 3]:
            if len(np_image.shape) == 2:
                np_image.shape = np_image.shape + (1,)
            elif len(np_image.shape) == 3:
                if np_image.shape[2] in [1, 3, 4]:
                    pass
                else:
                    return 4
            if (min_res < np_image.shape[0] < max_res) and (min_res < np_image.shape[1] < max_res):
                return cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
            else:
                return 9
        else:
            return 3
    else:
        return 2


def detect_faces(image_rgb: np.ndarray,
                 detection_method: str,
                 rotate: bool = False,
                 match_step: Optional[bool] = False) -> Union[Tuple[np.ndarray, int],
                                                              Tuple[np.ndarray, np.ndarray, tuple]]:
    """
    Function to detect faces in the given image. If no face detected then, it rotates the image(90)
     and then checks for faces in the rotated image
    :param image_rgb: rgb input image
    :param detection_method: cnn(high accuracy but slow) or hog(fast but low accuracy)
    :param rotate: whether to match by rotating image when no face is found
    :param match_step: boolean to detect face at matching stage
    :return: correct image(rotated if needed) and locations of boxes of detected faces
    """
    image_boxes = None
    rotated_image_rgb = image_rgb
    if rotate:
        for rotate_num in range(0, 4):
            image_boxes = face_recognition.face_locations(rotated_image_rgb, model=detection_method)
            if len(image_boxes) == 1:
                break
            elif len(image_boxes) > 1:
                return rotated_image_rgb, 1
            elif (not match_step) and rotate_num < 3:
                print('    No faces detected, rotating image by {} degree'.format((rotate_num+1)*90))
                rotated_image_rgb = imutils.rotate_bound(rotated_image_rgb, 90)
    else:
        image_boxes = face_recognition.face_locations(rotated_image_rgb, model=detection_method)
        if len(image_boxes) > 1:
            # sys.exit('Found {} faces in the image. Please check the image'.format(len(image_boxes)))
            return rotated_image_rgb, 1

    # check if no face is detected and return image_box = 0
    if len(image_boxes) == 0:
        return rotated_image_rgb, 0
    # return rotated images and the box locations if only one face is detected
    else:
        return rotated_image_rgb, image_boxes


def process_image(image: bytes, detection_method: str, rotate: Optional[bool] = False):
    """
    function to process the image: load image, detect face, find embeddings
    :param image: image in bytes data format
    :param detection_method: cnn(high accuracy but slow) or hog(fast but low accuracy)
    :param rotate: whether to process image by rotating it or not
    :return: rgb_image, bounding box location, face embeddings
    """
    # load the input image and convert it from RGB (OpenCV ordering) to dlib ordering (RGB)
    image_rgb = load_rgb_image(image)

    if type(image_rgb) == int:
        if image_rgb == 2:
            return 2, None, None
        elif image_rgb == 9:
            return 9, None, None
        elif image_rgb == 3:
            return 3, None, None
        elif image_rgb == 4:
            return 4, None, None
    # detect faces in the first image
    image_rgb, image_face_box = detect_faces(image_rgb, detection_method, rotate)
    # exit if no face is detected
    if image_face_box == 0:
        image_rgb, image_encodings = 6, None
        return image_rgb, image_face_box, image_encodings
    # exit if more than one face is detected
    elif image_face_box == 1:
        image_rgb, image_encodings = 7, None
        return image_rgb, image_face_box, image_encodings

    # Find encodings in the image
    image_encodings = face_recognition.face_encodings(image_rgb, image_face_box)
    return image_rgb, image_face_box, image_encodings


def rematch(image: bytes, first_image_encodings: list, detection_method: str) -> Tuple[Union[bool, int],
                                                                                       np.ndarray,
                                                                                       tuple]:
    """
    function to match faces by rotating second image when match fails on the given images
    :param image: second image in bytes data format
    :param first_image_encodings: first image encodings
    :param detection_method: cnn(high accuracy but slow) or hog(fast but low accuracy)
    :return: match(boolean), second image, bounding box locations
    """
    face_box = None
    image_rgb_copy = load_rgb_image(image)
    # rotate the second image three times and check each time if the face matches the face in first image
    for i in range(0, 3):
        print(' rotating second image by {} degree'.format((i+1)*90))
        # rotate the image by 90 degrees
        image_rgb = imutils.rotate_bound(image_rgb_copy, 90)
        image_rgb, face_box = detect_faces(image_rgb,
                                           detection_method,
                                           rotate=True,
                                           match_step=True)
        if face_box == 0:
            print(' count not detect face, trying again...')
            continue
        elif face_box == 1:
            print(' detected more than one face, trying again...')
            continue
        else:
            image_encodings = face_recognition.face_encodings(image_rgb, face_box)
            match = face_recognition.compare_faces(first_image_encodings,
                                                   image_encodings[0],
                                                   tolerance=0.54)
            match = match[0]
            # break out of the loop and return (1, rotated image, face loc) if match is True
            # else return (0, org_rgb image, None)
            if match:
                print('match', match)
                return 1, image_rgb, face_box
            else:
                print(' False match, trying again...')

    return 0, image_rgb_copy, face_box


def draw_box(image: np.ndarray, face_box: Tuple[int, int, int, int], color: Tuple[int, int, int]) -> None:
    """
    function to draw bounding boxes on detected faces
    :param image: original image on which box is drawn
    :param face_box: coordinates of bounding box points
    :param color: red(0,0,255) for no match, green(0,255,0) for match
    :return: None
    """
    for (top, right, bottom, left) in face_box:
        # draw the predicted face on the image
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)


def error_code(image_1: Union[int, np.ndarray], image_2: Union[int, np.ndarray]) -> Union[int, None]:
    """
    function to check if the images are valid and can be processed
    :param image_1: image 1 (either numpy array- if image or int - if error)
    :param image_2: image 2(either numpy array- if image or int - if error)
    :return: return None if valid and if not then return the error code configured in properties file
    """

    if type(image_1) == int:
        return image_1
    elif type(image_2) == int:
        return image_2
    return None


def show_images(image_1_rgb: np.ndarray, image_2_rgb: np.ndarray) -> None:
    """
    function to display the images side by side for debugging (only for internal purpose)
    :param image_1_rgb: first rgb image
    :param image_2_rgb: second rgb error
    :return: None
    """
    # resize the images and display side by side to compare
    resize_1 = cv2.resize(image_1_rgb, (600, 600))
    resize_2 = cv2.resize(image_2_rgb, (600, resize_1.shape[0]))
    # create a single numpy array by stacking the images side by side
    numpy_horizontal = np.hstack((resize_1, resize_2))
    cv2.imshow('Input images', numpy_horizontal)
    cv2.waitKey(0)
