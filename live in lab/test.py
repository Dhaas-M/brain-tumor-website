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
# from IPython.display import Image,display


#print("before")

# for n, a in enumerate(sys.argv):
#     print('arg {} has value {} endOfArg'.format(n, a))

path = str(sys.argv[1])

# model = load_model('BT_model_1.1.h5')
# batch_size = 10
#image = cv2.imread('/Users/91877/Desktop/nodeProjects/live in lab/public/uploads/'+path)
#Test('/Users/91877/Desktop/nodeProjects/live in lab/public/uploads/'+path)
# img = Image.fromarray(image)
# img = img.resize((300,300))
# img = np.array(img)
# input_img = np.expand_dims(img, axis=0)
# #result = model.predict_classes(input_img)
# result=model.predict(input_img) 
# print(result)

def Test(In_Img):
    #In_Img='pred16.jpg'

    #i = In_Img[23:34])

    #Img_Dis(In_Img)

    model = load_model('BT_model_1.1.h5')
    batch_size = 10
    image = cv2.imread(In_Img)
    img = Image.fromarray(image)
    img = img.resize((300,300))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    #result = model.predict_classes(input_img)
    result=model.predict(input_img) 
    print(result)

    #display(Image(input_img))
    res_path = generate()
    # print(res_path)
    #_Using_different_filters_to_show_images_in_various_views
    i1 = cv2.imread(In_Img)
    img = cv2.imread(In_Img)
    i = cv2.imread(In_Img)
    img = img - 100.000
    img_new = meijering(img)
    img1 =   meijering(img)
    img_1 =  rgbcie2rgb(img1) - 1 
    img2 = rgb2lab(img)
    # 
    plt.plot(),imshow(i1)
    plt.title("Input Image")
    plt.savefig('public/result/'+res_path+'1.png')
    print('result/'+res_path+'1.png')

    plt.subplot(121),imshow(i)
    plt.title("input_image")

    plt.subplot(122), imshow(img)
    plt.title("View-1")
    plt.savefig('public/result/'+res_path+'2.png')
    print('result/'+res_path+'2.png')

    plt.subplot(121), imshow(img2)
    plt.title("View-2")
    #plt.show()

    plt.subplot(122), imshow(img_1)
    plt.title("View-3")
    plt.savefig('public/result/'+res_path+'3.png')
    print('result/'+res_path+'3.png')

    #plt.subplot(121), imshow(img_new)
    #plt.title("View-4")
    # plt.title('RGB Format') 
    
    # plt.subplot(122), imshow(img_new)
    # plt.title('HSV Format') 
    # #Img_cr(In_Img)


    # model = load_model('Brain_T_Train.h5')
    # batch_size = 10
    # image = cv2.imread(In_Img)
    # img = Image.fromarray(image)
    # img = img.resize((300,300))
    # img = np.array(img)
    # input_img = np.expand_dims(img, axis=0)
    # #result = model.predict_classes(input_img)
    # result=model.predict(input_img) 
    # print(result)

Test('/Users/91877/Desktop/nodeProjects/live in lab/public/uploads/'+path)
#model = load_model('brain_tumor_model_1.h5')
# image = cv2.imread('/Users/91877/Desktop/nodeProjects/live in lab/public/uploads/'+path)
# img = Image.fromarray(image)
# img = img.resize((300,300))
# img = np.array(img)
# input_img = np.expand_dims(img, axis=0)
# result = model.predict(input_img)
# print(result)


#print(path)
