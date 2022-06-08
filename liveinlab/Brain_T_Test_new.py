# Test_Model_Performance
# __Start_Testing___....


# _Importing_Modules....
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
# from Brain_T_Display import Img_Dis
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from skimage.color import  rgbcie2rgb, rgb2lab
from skimage.filters import meijering
from keras import  Model
import joblib

# from Brain_T_Display import Img_Dis

# from Brain_T_color import Img_cr
# Function_for_performing_test

def Test(In_Img,ch):
    # In_Img='pred16.jpg'

    # i = In_Img[23:34])

    # Img_Dis(In_Img)
    if(ch==1):
        print("BRAIN STAGES")
        # _Using_different_filters_to_show_images_in_various_views
        i1 = imread(In_Img)
        img = imread(In_Img)
        i = imread(In_Img)
        img = img - 100.000
        # img_new = meijering(img)
        img1 = meijering(img)
        img_1 = rgbcie2rgb(img1) - 1
        img2 = rgb2lab(img)
        #
        plt.plot(), imshow(i1)
        plt.title("Input Image")
        plt.show()

        plt.subplot(121), imshow(i)
        plt.title("input_image")

        plt.subplot(122), imshow(img)
        plt.title("View-1")
        plt.show()

        plt.subplot(121), imshow(img2)
        plt.title("View-2")

        plt.subplot(122), imshow(img_1)
        plt.title("View-3")
        plt.show()

        # plt.subplot(121), imshow(img_new)
        # plt.title("View-4")
        # plt.title('RGB Format')

        # plt.subplot(122), imshow(img_new)
        # plt.title('HSV Format')
        # Img_cr(In_Img)
        model = load_model('D:\Brain-CNN.h5')
        batch_size = 10
        image = cv2.imread(In_Img)
        img = Image.fromarray(image)
        img = img.resize((100, 100))
        img = np.array(img)
        input_img = np.expand_dims(img, axis=0)
        # print(input_img)
        # print(input_img.shape)
        # result = model.predict_classes(input_img)
        result = model.predict(input_img)
        # print(result)
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.output)
        i = intermediate_layer_model.predict(input_img)
        # print("in_m",i)
        # Load the SVM_model from the file
        svm_model = joblib.load('D:\Brain-Conv-SVM.pkl')
        # Use the loaded model to make predictions
        r = svm_model.predict(i)
        # print(r[0])

        # Load the SVM_model from the file
        xgb_model = joblib.load('D:\Brain-Conv-XGB.pkl')
        # Use the loaded model to make predictions
        r1 = xgb_model.predict(i)
        # print(r1[0])
        # svm-model = load_model('D:\Conv-SVM.pkl')
        # r = svm-model.predict()
        # print(r)

        # print("TUMOR STATUS:")
        # Tumor_present_print_[[0]]
        # Tumor_absent_print_[[1]]
        if (result == 1):
            print("___/\___/\_____Tumour_Present_____/\______/\__")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (11, 11), 0)
            canny = cv2.Canny(blur, 10, 150, 3)
            dilated = cv2.dilate(canny, (1, 1), iterations=0)

            (cnt, hierarchy) = cv2.findContours(
                dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.drawContours(rgb, cnt, -1, (0, 256, 0), 0)

            print("___/\___/\_____Tumour_Present_____/\______/\__")
            print("Brain Stage: Abnormal")

            print("Tumor Trace: ", len(cnt))
            print("Take rest ")
        else:
            print("___/\___/\_____Tumour_Absent______/\______/\__")
            print("You_/\_are_/\_perfectly_/\_all-right")


    elif(ch==2):
        print("lung STAGES")
        # _Using_different_filters_to_show_images_in_various_views
        i1 = imread(In_Img)
        img = imread(In_Img)
        i = imread(In_Img)
        img = img - 100.000
        # img_new = meijering(img)
        img1 = meijering(img)
        img_1 = rgbcie2rgb(img1) - 1
        img2 = rgb2lab(img)
        #
        plt.plot(), imshow(i1)
        plt.title("Input Image")
        plt.show()

        plt.subplot(121), imshow(i)
        plt.title("input_image")

        plt.subplot(122), imshow(img)
        plt.title("View-1")
        plt.show()

        plt.subplot(121), imshow(img2)
        plt.title("View-2")

        plt.subplot(122), imshow(img_1)
        plt.title("View-3")
        plt.show()
        # plt.subplot(121), imshow(img_new)
        # plt.title("View-4")
        # plt.title('RGB Format')

        # plt.subplot(122), imshow(img_new)
        # plt.title('HSV Format')
        # Img_cr(In_Img)
        model = load_model('D:\lung-conv-SVM.h5')
        batch_size = 10
        image = cv2.imread(In_Img)
        img = Image.fromarray(image)
        img = img.resize((100, 100))
        img = np.array(img)
        input_img = np.expand_dims(img, axis=0)
        # print(input_img)
        # print(input_img.shape)
        # result = model.predict_classes(input_img)
        result = model.predict(input_img)
        # print(result)
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.output)
        i = intermediate_layer_model.predict(input_img)
        # print("in_m", i)
        # Load the SVM_model from the file
        svm_model = joblib.load('D:\lung-Conv-SVM.pkl')
        # Use the loaded model to make predictions
        r = svm_model.predict(i)
        # print(r[0])

        # Load the SVM_model from the file
        xgb_model = joblib.load('D:\lung-Conv-XGB.pkl')
        # Use the loaded model to make predictions
        r1 = xgb_model.predict(i)
        # print(r1[0])
        # svm-model = load_model('D:\Conv-SVM.pkl')
        # r = svm-model.predict()
        # print(r)

        # print("TUMOR STATUS:")
        # Tumor_present_print_[[0]]
        # Tumor_absent_print_[[1]]
        if (result == 1 and r==1 and r1==1):
            print("___/\___/\_____Tumour_Present_____/\______/\__")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (11, 11), 0)
            canny = cv2.Canny(blur, 10, 150, 3)
            dilated = cv2.dilate(canny, (1, 1), iterations=0)

            (cnt, hierarchy) = cv2.findContours(
                dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.drawContours(rgb, cnt, -1, (0, 256, 0), 0)

            print("___/\___/\_____Tumour_Present_____/\______/\__")
            print("Lung Stage: Abnormal")

            print("Tumor Trace: ", len(cnt))
            print("Take rest ")
        else:
            print("___/\___/\_____Tumour_Absent______/\______/\__")
            print("You_/\_are_/\_perfectly_/\_all-right")

    elif(ch==3):
        print("Kidney STAGES")
        # _Using_different_filters_to_show_images_in_various_views
        i1 = imread(In_Img)
        img = imread(In_Img)
        i = imread(In_Img)
        img = img - 100.000
        # img_new = meijering(img)
        img1 = meijering(img)
        img_1 = rgbcie2rgb(img1) - 1
        img2 = rgb2lab(img)
        #
        plt.plot(), imshow(i1)
        plt.title("Input Image")
        plt.show()

        plt.subplot(121), imshow(i)
        plt.title("input_image")

        plt.subplot(122), imshow(img)
        plt.title("View-1")
        plt.show()

        plt.subplot(121), imshow(img2)
        plt.title("View-2")
        plt.show()

        plt.subplot(122), imshow(img_1)
        plt.title("View-3")
        plt.show()



        model = load_model('D:\kidney-conv-SVM.h5')
        batch_size = 10
        image = cv2.imread(In_Img)
        img = Image.fromarray(image)
        img = img.resize((100, 100))
        img = np.array(img)
        input_img = np.expand_dims(img, axis=0)
        # print(input_img)
        # print(input_img.shape)
        # result = model.predict_classes(input_img)
        result = model.predict(input_img)
        # print(result)
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.output)
        i = intermediate_layer_model.predict(input_img)
        # print("in_m", i)
        # Load the SVM_model from the file
        svm_model = joblib.load('D:\kidney-Conv-SVM.pkl')
        # Use the loaded model to make predictions
        r = svm_model.predict(i)
        # print(r[0])

        # Load the SVM_model from the file
        xgb_model = joblib.load('D:\kidney-Conv-XGB.pkl')
        # Use the loaded model to make predictions
        r1 = xgb_model.predict(i)
        # print(r1[0])
        # svm-model = load_model('D:\Conv-SVM.pkl')
        # r = svm-model.predict()
        # print(r)

        # print("TUMOR STATUS:")
        # Tumor_present_print_[[0]]
        # Tumor_absent_print_[[1]]
        if (result == 1 and r==1 and r1==1):
            print("___/\___/\_____Tumour_Present_____/\______/\__")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (11, 11), 0)
            canny = cv2.Canny(blur, 10, 150, 3)
            dilated = cv2.dilate(canny, (1, 1), iterations=0)

            (cnt, hierarchy) = cv2.findContours(
                dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.drawContours(rgb, cnt, -1, (0, 256, 0), 0)

            print("___/\___/\_____Tumour_Present_____/\______/\__")
            print("Kidney Stage: Abnormal")

            print("Tumor Trace: ", len(cnt))
            print("Take rest ")
        else:
            print("___/\___/\_____Tumour_Absent______/\______/\__")
            print("You_/\_are_/\_perfectly_/\_all-right")

    # __Test__Ended


print("Tumor Detector")
print()

name = str(input("Enter the Name: "))
age = int(input("Enter your Age:"))
doc_name = str(input("Enter Doctor Name:"))
print("Enter the choice:")
print("1.Brain")
print("2.Lung")
print("3.kidney")
ch = int(input())
In_Img = input("Enter the Image:")
print()
print()
print()
print("______________________________________________________________________________________")
print('Name:', name)
print("Age:", age)
print("Doctor name:", doc_name)
print("Image")
print()
# Img_Dis(In_Img)
Test(In_Img,ch)

#
# import os
# print(os.listdir('C:/Users/YK/PycharmProjects/Py/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Tumor/Tumor- (1).jpg'))

