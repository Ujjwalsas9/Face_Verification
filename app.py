from flask import Flask, request, Response
# from flask import redirect, url_for, render_template
# from werkzeug.utils import secure_filename
from compare_faces_images import compare_faces
import jsonpickle
import base64 as b64
from json import loads
import configparser

app = Flask(__name__)


@app.route('/')
def home():
    return "<h1>Invalid Page</h1>"


@app.route('/compare-api/', methods=['POST'])
def uploaded_file():
    match_value = None
    mimetype = config.get('api', 'mimetype')
    img_json = loads(request.data)


    image_1 = b64.b64decode(img_json['image1'].encode())
    image_2 = b64.b64decode(img_json['image2'].encode())
    detection_method = img_json['detection_method']
    rotate_image = img_json['rotate_images']

    print(image_1, image_2)
    print("Image Got")

    match_value = compare_faces(image_1=image_1,
                                image_2=image_2,
                                detection_method=detection_method,
                                rotate_image2=rotate_image)
    print(match_value)
    print("Matched Val")
    return Response(response=jsonpickle.encode({"result": match_value}), status=200, mimetype=mimetype)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('service.properties')
    host = config.get('api', 'HOST')
    port = int(config.get('api', 'PORT'))
    app.run(debug=True,
            host=host,
            port=port)
