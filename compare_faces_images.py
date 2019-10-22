# USAGE
# python compare_faces_images.py --first-image images/saif.png --second-image images/saif_2.png
# --detection-method hog

# import required packages
import face_recognition
from image_prep import draw_box, process_image, rematch, error_code, show_images


def compare_faces(image_1: bytes,
                  image_2: bytes,
                  detection_method: str = 'hog',
                  rotate_image2: bool = False,
                  display_images: bool = False) -> int:
    """
    function to compare faces
    :param image_1: first image with a face in it
    :param image_2: second image with a face in it
    :param detection_method: either 'hog' or 'cnn'
    :param rotate_image2: If true then, rotate the second image to find the face and then
           compare it with face found in first image
    :param display_images: for internal (develeopers) use only. Whether to display the images side by side to
           compare them
    :return: Code(integer value) that is decoded to output appropriate message from the properties file
    """
    # load and process the first image
    image1_rgb, image1_face_box, image1_encodings = process_image(image=image_1,
                                                                  detection_method=detection_method,
                                                                  rotate=rotate_image2)

    # load and process the second image
    image2_rgb, image2_face_box, image2_encodings = process_image(image=image_2,
                                                                  detection_method=detection_method,
                                                                  rotate=rotate_image2)

    # check if both the images are processed correctly without encountering an error
    return_code = error_code(image1_rgb, image2_rgb)
    if return_code is not None:
        return return_code

    # check if the faces in both the images match (tolerance=dist between encodings, reduce it for strict match)
    matches = face_recognition.compare_faces(image1_encodings, image2_encodings[0], tolerance=0.54)

    # if there is a match then return_code = 1 else its 0
    return_code = 1 if matches[0] else 0

    # If no match and rotate parameter is True, then rotate the second image and then repeat the process
    if rotate_image2 and (not return_code):
        return_code, new_image2_rgb, new_image2_face_box = rematch(image=image_2,
                                                                   first_image_encodings=image1_encodings,
                                                                   detection_method=detection_method)
        # If there is a match then assign the rotated image & face box values to original second image and face_box
        if return_code:
            image2_rgb, image2_face_box = new_image2_rgb, new_image2_face_box

    # To verify the images display images side by side
    if display_images:
        # draw bounding box on face with green color
        if return_code:
            draw_box(image1_rgb, image1_face_box, color=(0, 255, 0))
            draw_box(image2_rgb, image2_face_box, color=(0, 255, 0))
        # draw bounding box on face with red color
        else:
            draw_box(image1_rgb, image1_face_box, color=(0, 0, 255))
            draw_box(image2_rgb, image2_face_box, color=(0, 0, 255))
        show_images(image_1_rgb=image1_rgb, image_2_rgb=image2_rgb)

    return return_code

