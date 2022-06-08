import sys 
import cv2
from nanoid import generate
from keras.models import load_model
from PIL import Image
import numpy as np
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from skimage.color import rgb2ypbpr,rgbcie2rgb,rgb2lab,rgb2hed
from skimage.filters import meijering
import numpy as np
import tensorflowjs as tfjs
import joblib
from keras import Model


path = str(sys.argv[1])
type = str(sys.argv[2])



def TestBT(In_Img):
        res_path = generate()
    # In_Img='pred16.jpg'

    # i = In_Img[23:34])

    # Img_Dis(In_Img)
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
        # plt.axis('off')
        plt.savefig('public/result/'+res_path+'1.png')
        print('result/'+res_path+'1.png')

        plt.subplot(121), imshow(i)
        plt.title("input_image")
        # plt.axis('off')

        plt.subplot(122), imshow(img)
        plt.title("View-1")
        # plt.axis('off')
        plt.savefig('public/result/'+res_path+'2.png')
        print('result/'+res_path+'2.png')

        plt.subplot(121), imshow(img2)
        plt.title("View-2")

        plt.subplot(122), imshow(img_1)
        plt.title("View-3")
        # plt.axis('off')
        plt.savefig('public/result/'+res_path+'3.png')
        print('result/'+res_path+'3.png')

        # plt.subplot(121), imshow(img_new)
        # plt.title("View-4")
        # plt.title('RGB Format')

        # plt.subplot(122), imshow(img_new)
        # plt.title('HSV Format')
        # Img_cr(In_Img)
        model = load_model('Brain-CNN.h5')
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
        svm_model = joblib.load('Brain-Conv-SVM.pkl')
        # Use the loaded model to make predictions
        r = svm_model.predict(i)
        # print(r[0])

        # Load the SVM_model from the file
        xgb_model = joblib.load('Brain-Conv-XGB.pkl')
        # Use the loaded model to make predictions
        r1 = xgb_model.predict(i)
        print(result,r,r1)
        # print(r1[0])
        # svm-model = load_model('D:\Conv-SVM.pkl')
        # r = svm-model.predict()
        # print(r)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        canny = cv2.Canny(blur, 10, 150, 3)
        dilated = cv2.dilate(canny, (1, 1), iterations=0)

        (cnt, hierarchy) = cv2.findContours(
            dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.drawContours(rgb, cnt, -1, (0, 256, 0), 0)
        print(len(cnt))
        # print("TUMOR STATUS:")
        # Tumor_present_print_[[0]]
        # Tumor_absent_print_[[1]]

def TestKT(In_Img):
        res_path = generate()
    # In_Img='pred16.jpg'

    # i = In_Img[23:34])

    # Img_Dis(In_Img)
        print("KIDNEY STAGES")
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
        # plt.axis('off')
        plt.savefig('public/result/'+res_path+'1.png')
        print('result/'+res_path+'1.png')

        plt.subplot(121), imshow(i)
        plt.title("input_image")
        # plt.axis('off')

        plt.subplot(122), imshow(img)
        plt.title("View-1")
        # plt.axis('off')
        plt.savefig('public/result/'+res_path+'2.png')
        print('result/'+res_path+'2.png')

        plt.subplot(121), imshow(img2)
        plt.title("View-2")

        plt.subplot(122), imshow(img_1)
        plt.title("View-3")
        # plt.axis('off')
        plt.savefig('public/result/'+res_path+'3.png')
        print('result/'+res_path+'3.png')

        # plt.subplot(121), imshow(img_new)
        # plt.title("View-4")
        # plt.title('RGB Format')

        # plt.subplot(122), imshow(img_new)
        # plt.title('HSV Format')
        # Img_cr(In_Img)
        model = load_model('kidney-conv-SVM.h5')
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
        svm_model = joblib.load('kidney-conv-SVM.pkl')
        # Use the loaded model to make predictions
        r = svm_model.predict(i)
        # print(r[0])

        # Load the SVM_model from the file
        xgb_model = joblib.load('kidney-conv-SVM.pkl')
        # Use the loaded model to make predictions
        r1 = xgb_model.predict(i)
        print(result,r,r1)
        # print(r1[0])
        # svm-model = load_model('D:\Conv-SVM.pkl')
        # r = svm-model.predict()
        # print(r)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        canny = cv2.Canny(blur, 10, 150, 3)
        dilated = cv2.dilate(canny, (1, 1), iterations=0)

        (cnt, hierarchy) = cv2.findContours(
            dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.drawContours(rgb, cnt, -1, (0, 256, 0), 0)
        print(len(cnt))
        # print("TUMOR STATUS:")
        # Tumor_present_print_[[0]]
        # Tumor_absent_print_[[1]]  


def TestLT(In_Img):
        res_path = generate()
    # In_Img='pred16.jpg'

    # i = In_Img[23:34])

    # Img_Dis(In_Img)
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
        # plt.axis('off')
        plt.savefig('public/result/'+res_path+'1.png')
        print('result/'+res_path+'1.png')

        plt.subplot(121), imshow(i)
        plt.title("input_image")
        # plt.axis('off')

        plt.subplot(122), imshow(img)
        plt.title("View-1")
        # plt.axis('off')
        plt.savefig('public/result/'+res_path+'2.png')
        print('result/'+res_path+'2.png')

        plt.subplot(121), imshow(img2)
        plt.title("View-2")

        plt.subplot(122), imshow(img_1)
        plt.title("View-3")
        # plt.axis('off')
        plt.savefig('public/result/'+res_path+'3.png')
        print('result/'+res_path+'3.png')

        # plt.subplot(121), imshow(img_new)
        # plt.title("View-4")
        # plt.title('RGB Format')

        # plt.subplot(122), imshow(img_new)
        # plt.title('HSV Format')
        # Img_cr(In_Img)
        model = load_model('lung-conv-SVM.h5')
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
        svm_model = joblib.load('lung-Conv-SVM.pkl')
        # Use the loaded model to make predictions
        r = svm_model.predict(i)
        # print(r[0])

        # Load the SVM_model from the file
        xgb_model = joblib.load('lung-Conv-XGB.pkl')
        # Use the loaded model to make predictions
        r1 = xgb_model.predict(i)
        print(result,r,r1)
        # print(r1[0])
        # svm-model = load_model('D:\Conv-SVM.pkl')
        # r = svm-model.predict()
        # print(r)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        canny = cv2.Canny(blur, 10, 150, 3)
        dilated = cv2.dilate(canny, (1, 1), iterations=0)

        (cnt, hierarchy) = cv2.findContours(
            dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.drawContours(rgb, cnt, -1, (0, 256, 0), 0)
        print(len(cnt))
        # print("TUMOR STATUS:")
        # Tumor_present_print_[[0]]
        # Tumor_absent_print_[[1]]              



# def Test(In_Img):
#     #In_Img='pred16.jpg'

#     #i = In_Img[23:34])

#     #Img_Dis(In_Img)

#     model = load_model('BT_model_1.1.h5')
#     batch_size = 10
#     image = cv2.imread(In_Img)
#     img = Image.fromarray(image)
#     img = img.resize((300,300))
#     img = np.array(img)
#     input_img = np.expand_dims(img, axis=0)
#     #result = model.predict_classes(input_img)
#     result=model.predict(input_img) 
#     print(result)

#     #display(Image(input_img))
#     res_path = generate()
#     # print(res_path)
#     #_Using_different_filters_to_show_images_in_various_views
#     i1 = cv2.imread(In_Img)
#     img = cv2.imread(In_Img)
#     i = cv2.imread(In_Img)
#     img = img - 100.000
#     img_new = meijering(img)
#     img1 =   meijering(img)
#     img_1 =  rgbcie2rgb(img1) - 1 
#     img2 = rgb2lab(img)
#     # 
#     plt.plot(),imshow(i1)
#     plt.title("Input Image")
#     plt.savefig('public/result/'+res_path+'1.png')
#     print('result/'+res_path+'1.png')

#     plt.subplot(121),imshow(i)
#     plt.title("input_image")

#     plt.subplot(122), imshow(img)
#     plt.title("View-1")
#     plt.savefig('public/result/'+res_path+'2.png')
#     print('result/'+res_path+'2.png')

#     plt.subplot(121), imshow(img2)
#     plt.title("View-2")
#     #plt.show()

#     plt.subplot(122), imshow(img_1)
#     plt.title("View-3")
#     plt.savefig('public/result/'+res_path+'3.png')
#     print('result/'+res_path+'3.png')

if type == 'BT':TestBT('/Users/91877/Desktop/brain-tumor-website/liveinlab/public/uploads/'+path)
if type == 'KT':TestKT('/Users/91877/Desktop/brain-tumor-website/liveinlab/public/uploads/'+path)
if type == 'LT':TestLT('/Users/91877/Desktop/brain-tumor-website/liveinlab/public/uploads/'+path)

# Test('/Users/91877/Desktop/brain-tumor-website/liveinlab/public/uploads/'+path)
