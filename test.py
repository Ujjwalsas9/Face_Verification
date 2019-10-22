import os
from compare_faces_images import compare_faces
import csv

test_data_path = '/home/xmplar/xfact/FaceIDAdrian/VGGFace2Data/train'
folders = sorted(os.listdir(test_data_path))

result = []
results_xl = os.path.join("/home/xmplar/xfact/FaceIDAdrian/VGGFace2Data", 'results.csv')
with open(results_xl, 'w') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(['Folder name', 'Total Images', 'Match', 'No match', 'No face', 'Multiple Face', 'Resolution', 'Accuracy'])

for i, folder in enumerate(folders):
    images_in_folder = sorted(os.listdir(os.path.join(test_data_path, folder)))
    total_images_in_folder = len(images_in_folder)
    res = []
    log_dict = {}
    first_image = images_in_folder[0]
    with open(os.path.join(test_data_path, folder, first_image), 'rb') as f1:
        first_image = f1.read()

    for image_name in images_in_folder:
        with open(os.path.join(test_data_path, folder, image_name), 'rb') as f1:
            image = f1.read()
        res.append(compare_faces(image_1=first_image, image_2=image, detection_method='cnn'))
        # if res[-1] == 6:
        #     print(image_name)
    match_value = res.count(1)
    compare_done_on = (res.count(1) + res.count(0))
    result.append((match_value * 100 / compare_done_on))

    log_dict['Folder name'] = folder
    log_dict['Total Images'] = total_images_in_folder
    log_dict['Match'] = res.count(1)
    log_dict['No match'] = res.count(0)
    log_dict['No face'] = res.count(6)
    log_dict['Multiple Face'] = res.count(7)
    log_dict['Resolution'] = res.count(9)
    log_dict['Accuracy'] = (match_value*100 / compare_done_on)

    with open(results_xl, 'a') as f:
        w = csv.DictWriter(f, (log_dict.keys()))
        w.writerow(log_dict)

#     print("\nTotal images in {} is {}".format(folder, total_images_in_folder))
#     print(" No Match", res.count(0))
#     print(" Match", res.count(1))
#     print(" No Face", res.count(6))
#     print(" Multiple Face", res.count(7))
#     print(" Resolution", res.count(9))
#     # with open(results_xl, 'a') as outcsv:
#     #     pass
# print('\n\n', result)
