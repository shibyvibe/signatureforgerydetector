#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications  ##Tensor flow version used 2.4.1
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet


# <b>Utility functions</b>

# In[71]:


def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_png(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, imageDimensions[:-1])
    return image

def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """
    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )

#Compare new signature speciment against all other positives
def getPersonEvaluationData(personId, path_SignatureSpecimen):
    
    person_df = datastore[datastore.personId == personId]
    
    anchor_imgs = []
    postive_imgs = []
    toCompare_imgs = []

    anchor_img = basePath + person_df.iloc[0].relPath + "/"+ person_df.iloc[0].fileName  
    for g in person_df[1:].index:
        pos_img= basePath+"/"+person_df.loc[g].relPath + "/"+person_df.loc[g].fileName  
        toCompareImg = path_SignatureSpecimen 
        anchor_imgs.append(anchor_img)
        postive_imgs.append(pos_img)
        toCompare_imgs.append(toCompareImg)

    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_imgs)
    positive_dataset  = tf.data.Dataset.from_tensor_slices(postive_imgs)
    toCompare_imgs_dataset  = tf.data.Dataset.from_tensor_slices(toCompare_imgs)

    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, toCompare_imgs_dataset))
    dataset = dataset.map(preprocess_triplets)
    
    return dataset        

def getEmbeddings(row):
    anchor, positive, toCcompare = row
    return (
        embedding(resnet.preprocess_input(anchor)),
        embedding(resnet.preprocess_input(positive)),
        embedding(resnet.preprocess_input(toCcompare)),
    )

def getEmbeddingDataFrame(dataset):
    dataset = dataset.batch(1, drop_remainder=False)
    embedding_data = [getEmbeddings(row) for row in iter(dataset)]
    embeddings_data_df = pd.DataFrame(columns=["anchor_embedding", "positive_embedding","toCompare_embedding"], data=embedding_data)
    return embeddings_data_df

def compareEmbeds_row_kMeansdistance (anchor_embedding, positive_embedding, toCompare_embedding):
        ap_distance = tf.reduce_sum(tf.square(anchor_embedding - positive_embedding), -1)
        acompare_distance = tf.reduce_sum(tf.square(anchor_embedding - toCompare_embedding), -1)
        return (ap_distance, acompare_distance, (ap_distance-acompare_distance))

def compareEmbeds(df):
    df2 = df.apply(lambda row: compareEmbeds_row_kMeansdistance(row.anchor_embedding, row.positive_embedding, row.toCompare_embedding), axis=1,  result_type='expand')
    df2.columns = ["Pos", "ToCompare", "Diff"]
    df[["Pos", "ToCompare", "Diff"]]=df2[["Pos", "ToCompare", "Diff"]]
    return df

def isForgery(personId, path_SignatureSpecimen):
    #Prepares the evaluation data frame data for the provided person.
    dataset = getPersonEvaluationData(personId, path_SignatureSpecimen)
    
    #Get embeddings for genuine and tocompare signatures
    #Use one of the genuines as anchor and get the distances between the remaining genuines and to compare specimen from anchor
    embeddings_df = getEmbeddingDataFrame(dataset)
    df_embedding_distances = compareEmbeds(embeddings_df)
     
    # Add column person Id to dataframe
    df_embedding_distances["personId"]  = personId
    
    y_predict = lr.predict(df_embedding_distances[["personId","Pos", "ToCompare"]])
    #y_tst_predict = lr.predict(df_test_forgery[["personId", "Pos", "Neg"]])  #neg = tocompare
    
    # Result % probability
    return "Probablity of forgery {}%".format(sum(y_predict)/len(y_predict)*100)


# <b> Load data from Datastore </b>

# In[32]:


basePath = "/notebooks/capstone/dataset/dataset2/sign_data/test/"
datastore = pd.read_csv(basePath + "/test_clean.csv")
datastore = datastore[datastore.Genuine == 1]
datastore.sort_values(by="personId", inplace=True)

imageDimensions = (224,224,3)


# <b>Set Model load parameters</b>

# In[33]:


model_embeddings_file = '/notebooks/capstone/models/embeddings-res32'       ##DNN Embeddings models save location
embedding = tf.keras.models.load_model(model_embeddings_file)


# In[34]:


pFile_lrmodel = "/notebooks/capstone/train_lr_k.pickle"  #squared distances
with open(pFile_lrmodel, 'rb') as file:
    lr = pickle.load(file)


# <b>Test api</b>

# In[73]:


f = isForgery(51, '/notebooks/capstone/dataset/dataset2/sign_data/test/051/06_051.png')  #Good
print("Forgery========>" + f )

# In[72]:


f = isForgery(49, '/notebooks/capstone/dataset/dataset2/sign_data/test/049_forg/01_0114049.PNG')  #forged
print("Forgery========>" + f )

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[74]:


# def visualize(images_dataset):
    
#     def showImages(ax, image):
#         for i in range(3):
#             ax[i].imshow(image[i])
#             ax[i].get_xaxis().set_visible(False)
#             ax[i].get_yaxis().set_visible(False)
#     rows = len(images_dataset)
#     images = list(images_dataset.as_numpy_iterator())
#     fig, axs = plt.subplots(rows,3, sharex=True, sharey=True, figsize=(30,30))
#     for x in range(rows):
#          anchor, positive, negative = images[x][0],images[x][1],images[x][2]
#          showImages(axs[x], (anchor, positive, negative))


# In[75]:


# temp_dataset = getPersonEvaluationData(51, '/notebooks/capstone/dataset/dataset2/sign_data/test/051/06_051.png')
# visualize(temp_dataset)

