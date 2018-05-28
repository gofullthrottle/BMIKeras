
# ### Usual Import

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

import os
import sys
import random
import json
import math
sys.path.append( root + 'Utils/')

import pandas as pd
import numpy as np
import tensorflow as tf

from PIL import Image
from IPython.display import display
from pprint import pprint
from notebook_utils import *

data_root = root + 'Datasets/BMIData/'

### Read Data


with open(data_root + 'wide_data.json') as f:
  data = json.load(f)
print len(data)

all_items = data.items()

train_data = dict( [all_items[i] for i in range(len(all_items)) if all_items[i][1]['is_training'] == 1] )
test_data =  dict( [all_items[i] for i in range(len(all_items)) if all_items[i][1]['is_training'] == 0] )

print len(train_data.items())
print len(test_data.items())


# ### Data Preparation
def get_data_by_id(id):
  img = Image.open(data[id]['path']).convert('L')
  max_edge = max(img.size[0], img.size[1])
  black = Image.new('L',(max_edge, max_edge),0)
  black.paste(img, [max_edge/2 - img.size[0]/2,
                    max_edge/2 - img.size[1]/2,
                    max_edge/2 + (img.size[0]+1)/2,
                    max_edge/2 + (img.size[1]+1)/2] )
  img = black.resize((224,224))
  img = np.array(img).astype(np.float32)
  img -= np.mean(img)
  img /= np.std( img )

  bmi = float(data[id]['bmi'])
  return img, bmi

image, bmi = get_data_by_id('1')
plt.imshow(image,cmap='Greys_r')
print bmi


# In[ ]:

def get_batch(id_list):
  size = len(id_list)
  image_batch = np.zeros((size,224,224,3))
  bmi_batch = np.zeros((size))

  for i in range(size):
    image, bmi = get_data_by_id( id_list[i] )
    image_batch[i,:,:,0] = image
    image_batch[i,:,:,1] = image
    image_batch[i,:,:,2] = image
    bmi_batch[i] = bmi
  return image_batch, bmi_batch

image_batch, bmi_batch = get_batch(['1534'])
plt.imshow(image_batch[0,:,:,0], cmap = 'Greys_r')
print bmi_batch[0]


#Construct TensorFlow Graph
sys.path.append(root + 'VGG_Face/')
sys.path.append('/afs/csail.mit.edu/u/k/Utku/Desktop/caffe-tensorflow-master/')
from VGG_Face import VGG_Face

image_ = tf.placeholder(tf.float32, shape = [None,224,224,3])
net = VGG_Face({'data' : image_}, trainable = True)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
net.load(root + 'VGG_Face/VGG_Face.npy',sess)


# ### Get the Train Embeddings
train_keys = train_data.keys()
train_embeddings = np.zeros((len(train_keys), 4096))
train_bmis = np.zeros((len(train_keys)))

for i,key in enumerate(train_keys):
  write(i)
  image_batch, bmi_batch = get_batch( [key] )
  embeddings = sess.run( net.layers['fc6'], feed_dict = {image_: image_batch})
  train_embeddings[i] = embeddings.flatten()
  train_bmis[i] = bmi_batch[0]

### Get the Test Embeddings

test_keys = test_data.keys()
test_embeddings = np.zeros((len(test_keys), 4096))
test_bmis = np.zeros((len(test_keys)))

for i,key in enumerate(test_keys):
  write(i)
  image_batch, bmi_batch = get_batch([key])
  embeddings = sess.run( net.layers['fc6'], feed_dict = {image_: image_batch})
  test_embeddings[i] = embeddings.flatten()
  test_bmis[i] = bmi_batch[0]


### Fit SVM

from sklearn.svm import SVR
import pickle

clf = SVR(C=0.1, kernel='linear', verbose = True)
clf.fit( np.concatenate([train_embeddings, test_embeddings]), np.concatenate([train_bmis, test_bmis]) )
with open('/afs/csail.mit.edu/u/k/Utku/Qatar/bmi_svm_regression_model.npy','wb') as f:
  pickle.dump( clf, f )
