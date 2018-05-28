from PIL import Image
from keras.preprocessing import image
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace
from sklearn.model_selection import train_test_split
import numpy as np
import glob
import pandas as pd
import csv
import pickle

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model,load_model
#4204,40.4928,Male,0,img_4204.bmp
path = 'img_4204.bmp'
img = Image.open('Release/Data/Images/' + path ).convert('L')
max_edge = max(img.size[0], img.size[1])
print(img.size)
image_sum = img.size[0] + img.size[1]
black = Image.new('L',(max_edge, max_edge),0)
crop_img = [int(max_edge/2 - img.size[0]/2) , int(max_edge/2 - img.size[1]/2),
            int(max_edge/2 + (img.size[0]+1)/2) ,int(max_edge/2 + (img.size[1]+1)/2)]


if (image_sum + crop_img[0]+crop_img[1] != crop_img[2] +crop_img[3]):
    if (crop_img[0] == 0):
        black.paste(img, [crop_img[0],crop_img[1]+1, crop_img[2],crop_img[3]])
    else:
        black.paste(img, [crop_img[0]+1,crop_img[1], crop_img[2],crop_img[3]])
else:
    black.paste(img, [crop_img[0],crop_img[1], crop_img[2],crop_img[3]])

    img = black.resize((224,224))
img = np.array(img).astype(np.float32)

img -= np.mean(img)
img /= np.std(img )
img = img.reshape((224,224,1))
img = np.concatenate([img,img,img],axis = 2)
img = img.reshape((1,224,224,3))
x = preprocess_input(img)


base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
fc6_pool_features = model.predict(x)
embeddings = fc6_pool_features.flatten()
print(embeddings.shape)
from sklearn.externals import joblib
embeddings = np.reshape(embeddings, (1,4096))
svr_path = 'bmi_svm_regression_model.pkl'

"""clf = joblib.load( 'bmi_svm_regression_model.pkl')
bmi_predict  = clf.predict(embeddings) 
print(bmi_predict)"""
"""clf = joblib.load( 'bmi_ridge_regression_model.pkl')
bmi_predict  = clf.predict(embeddings) 
print(bmi_predict)

"""
clf = joblib.load( 'bmi_krr_regression_model.pkl')
bmi_predict  = clf.predict(embeddings) 
print(bmi_predict)

