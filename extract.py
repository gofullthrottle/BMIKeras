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
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.svm import SVR
import pickle
from keras.models import model_from_json
from keras.models import load_model
from sklearn.externals import joblib

###################################### Load Data ##########################################################################
bmi_file = pd.read_csv("Release\Data\data.csv")
bmi_file['id'] = bmi_file.index
bmi = bmi_file['bmi']

i = 0
X_train_key = []
X_test_key  =[]
for index, row in bmi_file.iterrows():
    #print(row)
    t = row['name']
    #print(type(t))
    if row['is_training'] == 1:
        
        X_train_key.append(t)
    else:
        X_test_key.append(t)
 

################################### Data Preperation ##########################################################################

def get_data_by_id(path,bmi_file):
    #print(path)
    try:
        img = Image.open('Release/Data/Images/' + path ).convert('L')
        max_edge = max(img.size[0], img.size[1])
        #print(img.size)
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
        img /= np.std(img)

        bmi = bmi_file[bmi_file['name'] == path]['bmi']
        print(path)
        return img, bmi
    except ValueError:
        print('error')
            #print (type(bmi))
    
#image, bmi = get_data_by_id('0',bmi_file)
#print(image.shape)

def get_batch(id_list,bmi_file):

    size = len(id_list)
    image_batch = np.zeros((size,224,224,3))
    bmi_batch = np.zeros((size))

    for i in range(size):
        #print('id_list',id_list[i])
        image, bmi = get_data_by_id(id_list[i],bmi_file)
        image_batch[i,:,:,0] = image
        image_batch[i,:,:,1] = image
        image_batch[i,:,:,2] = image
        #print(i, type(i))
    #print(bmi, type(bmi))
    bmi_batch[i] = bmi
    return image_batch, bmi_batch

#image_batch, bmi_batch = get_batch(['1534'],bmi_file)
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

train_embeddings = np.zeros((len(X_train_key), 4096))
train_bmis = np.zeros((len(X_train_key)))
#len(X_train_key[406:]
for i in range(0,len(X_train_key)):
    #print('key initial', key)
    key = X_train_key[i]
    #key = 'img_408.bmp'
    print(key)
    #print(len(X_train_key[:4]))
    try:
        imag, bmi_batch =  get_batch([key], bmi_file)
        #print('image shape',imag.shape)
        x = preprocess_input(imag)
        fc6_pool_features = model.predict(x)
        train_embeddings[i] = fc6_pool_features.flatten()
        #print('embed', train_embeddings[i])
        train_bmis[i] = bmi_batch
    except TypeError:
        continue

    #print(train_bmis[i])
test_embeddings = np.zeros((len(X_test_key), 4096))
test_bmis = np.zeros((len(X_test_key)))

for i in range(0,len(X_test_key)):
    #print('key initial', key)
    key = X_test_key[i]
    try:
        imag, bmi_batch =  get_batch([key], bmi_file)
        #print('batch shape',imag.shape)
        
        x = preprocess_input(imag)
        fc6_pool_features = model.predict(x)
        test_embeddings[i] = fc6_pool_features.flatten()
        #print(len(train_embeddings[i]))
        test_bmis[i] = bmi_batch
    except TypeError:
        continue

### Fit SVR
print(train_embeddings[0], len(train_embeddings[0]))
clf = SVR(C=0.1, kernel='linear', verbose = True)
clf.fit( np.concatenate([train_embeddings, test_embeddings]), np.concatenate([train_bmis, test_bmis]) )
joblib.dump(clf, 'bmi_svm_regression_model.pkl') 




"""with open('bmi_svm_regression_model.h5','wb') as f:
    pickle.dump( clf, f )
f.close()"""