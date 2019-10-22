"""
This code is to test face_verification project.
It saves the names of different images and their comparison results in csv file.
It also saves the accuracy, precision, recall and f-score of the tests conducted in a separate text file.
"""

import os
from compare_faces_images import compare_faces
import csv

test_data_path_1 = '/home/FaceIDAdrian/VGGFace2Data/New_Image_Test_Data/African/Male_Different'
test_data_path_2 = '/home/FaceIDAdrian/VGGFace2Data/Test_Data_With_Different_ImagePair'
test_data_path_3 = '/home/FaceIDAdrian/VGGFace2Data/Test_Data_With_Human_ObjectPair'

"""
Make test_data_path = test_data_path_1 for comparing "Same" images and
test_data_path = test_data_path_2 for comparing "Different" images and
test_data_path = test_data_path_3 for comparing "Object" images.
"""

test_data_path = test_data_path_1
folders = sorted(os.listdir(test_data_path))

result = []
if test_data_path == test_data_path_1:
    results_xl = os.path.join(os.getcwd(), '/home/FaceIDAdrian/VGGFace2Data/New_Image_Test_Data/African/Male_Different/results_different.csv')
elif test_data_path == test_data_path_2:
    results_xl = os.path.join(os.getcwd(), 'results_different.csv')
elif test_data_path == test_data_path_3:
    results_xl = os.path.join(os.getcwd(), 'results_objects.csv')


with open(results_xl, 'w') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(['Folder name', 'Image_1', 'Image_2', 'Matched'])
res = []
log_dict = {}
for folder in folders:

    images_in_folder = os.listdir(os.path.join(test_data_path, folder))
    image1 = os.path.join(test_data_path, folder, images_in_folder[0])
    image2 = os.path.join(test_data_path, folder, images_in_folder[1])
    with open(image1, 'rb') as f1:
        first_image = f1.read()
    with open(image2, 'rb') as f2:
        second_image = f2.read()
    return_value = compare_faces(image_1=first_image, image_2=second_image, detection_method='cnn', display_images= True)
    res.append(return_value)
    log_dict['Folder name'] = folder
    log_dict['Image_1'] = images_in_folder[0]
    log_dict['Image_2'] = images_in_folder[1]
    log_dict['return_code'] = return_value
    with open(results_xl, 'a') as f:
        w = csv.DictWriter(f, (log_dict.keys()))
        w.writerow(log_dict)

if test_data_path == test_data_path_1:
    match_value = res.count(1)
elif test_data_path == test_data_path_2:
    match_value = res.count(0)
elif test_data_path == test_data_path_3:
    match_value = res.count(6)
compare_done_on = (res.count(1) + res.count(0))
if compare_done_on > 0:
    result.append((match_value * 100 / compare_done_on))

if test_data_path == test_data_path_1:
    true_positive = res.count(1)
    false_negative = res.count(0)
    true_negative = 0
    false_positive = 0
    precision = (true_positive/(true_positive+false_positive))
    recall = (true_positive/(true_positive+false_negative))
    f_score = 2 * ((precision * recall)/(precision + recall))
    # Accuracy => Classified as "Same"/Actually "Same"
    Accuracy_var = "Test Accuracy on same images is: " + str("%.2f" % round(result[-1], 2)) + "%" "\n"
    # Precision => Proportion of images we classified as same were actually same.
    Precision_var = "Precision on the same images is: " + str("%.2f" % round(precision, 2)) + "\n"
    # Recall => Proportion of images that actually are same, were classified as same.
    Recall_var = "Recall on the same images is: " + str("%.2f" % round(recall, 2)) + "\n"
    # Precision, also called the positive predictive value, is the proportion of positive results that truly are
    # positive. Recall, also called sensitivity, is the ability of a test to correctly identify positive results to
    # get the true positive rate. The F score reaches the best value, meaning perfect precision and recall,
    # at a value of 1. The worst F score, which means lowest precision and lowest recall, would be a value of 0.
    F_score_var = "F_score on the same images is: " + str("%.2f" % round(f_score, 2)) + "\n"
    file1 = open("result_for_same_images.txt", "w")
    file1.writelines(Accuracy_var)
    file1.writelines(Precision_var)
    file1.writelines(Recall_var)
    file1.writelines(F_score_var)
    file1.close()
elif test_data_path == test_data_path_2:
    true_positive = res.count(0)
    false_negative = res.count(1)
    true_negative = 0
    false_positive = 0
    precision = (true_positive / (true_positive + false_positive))
    recall = (true_positive / (true_positive + false_negative))
    f_score = 2 * ((precision * recall) / (precision + recall))
    # Accuracy => Classified as "Different"/Actually "Different"
    Accuracy_var = "Test Accuracy on different images is: " + str("%.2f" % round(result[-1], 2)) + "%" "\n"
    # Precision => Proportion of images we classified as different were actually different.
    Precision_var = "Precision on the different images is: " + str("%.2f" % round(precision, 2)) + "\n"
    # Recall => Proportion of images that actually are different, were classified as different.
    Recall_var = "Recall on different images is: " + str("%.2f" % round(recall, 2)) + "\n"
    # Precision, also called the positive predictive value, is the proportion of positive results that truly are
    # positive. Recall, also called sensitivity, is the ability of a test to correctly identify positive results to
    # get the true positive rate. The F score reaches the best value, meaning perfect precision and recall,
    # at a value of 1. The worst F score, which means lowest precision and lowest recall, would be a value of 0.
    F_score_var = "F_score on different images is: " + str("%.2f" % round(f_score, 2)) + "\n"
    file1 = open("result_for_different_images.txt", "w")
    file1.writelines(Accuracy_var)
    file1.writelines(Precision_var)
    file1.writelines(Recall_var)
    file1.writelines(F_score_var)
    file1.close()
