from json import dumps, loads
import base64 as b64
import requests
import configparser as cp

# Custom validation fn
from validation import file_type_validation


# Configuration file loading
config = cp.ConfigParser()
config.read('application.properties', encoding='utf-8')
IP_ADDRESS = config.get('Interface', 'IP_ADDRESS')
content_type = config.get('Interface', 'content_type')
headers = {'content-type': content_type}


def message_code(res):
    """

    :param res:
    :return:
    """
    message = config.get('messages', str(res))
    return message


file1, file2 = './../images/test1.png', './../images/kb.jpg'
validate = file_type_validation(file1, file2)

if validate == 'success':
    with open(file1, 'rb') as f1:
        img1 = f1.read()
    with open(file2, 'rb') as f2:
        img2 = f2.read()


    img_dict = {
                'image1': b64.b64encode(img1).decode('utf-8'),
                'image2': b64.b64encode(img2).decode('utf-8'),
                'detection_method': 'cnn',
                'rotate_images': False
                }
    imgJSON = dumps(img_dict)
    try:
        urlResponse = requests.post(IP_ADDRESS, data=imgJSON, headers=headers)
        result = loads(urlResponse.content.decode('utf-8'))['result']
        print(message_code(result))

    except Exception as e:
        print("Could not fetch data")
        print(e)
else:
    print(validate)
