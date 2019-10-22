from PIL import Image
import face_recognition

path_to_read_images = '/home/xmplar/xfact/FaceIDAdrian/VGGFace2Data/Object_Images_Original'
path_to_save_images = '/home/xmplar/xfact/FaceIDAdrian/VGGFace2Data/Object_Images_with_bounding_box'

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("/home/xmplar/xfact/FaceIDAdrian/VGGFace2Data/Object_Images_Original/ccc_600_lq_0001.jpg")

# Find all the faces in the image using the default HOG-based model.
# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
# See also: find_faces_in_picture_cnn.py
face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

print("I found {} face(s) in this photograph.".format(len(face_locations)))

for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()