#!/usr/bin/env python
# coding: utf-8

# <b> Approach </b>
# 
# 1) Use Deep Neural networks - Generate embeddings for signatures genuine and forgery use a triplet loss mining approach to separate the features of forged signatures from genuine.
# 2) Use Logistic regression for decisioning - use the embedding outputs from step1 and use these as features to train a logistics regression model to output the decision of genuine or forgery.
#    * A Deep Neural Network Model for decision would have worked better 

# <b style="color: red;">TODO </b>
# 1) Add signatures from dataset1 to dataset2 mainly related to background color <br>
# 2) Understand how to setup tensorflow GPU. Currently using Tensorflow CPU
# 

# <b>Tuning </b><p>
# To try
# * _compute_loss_sq change margin from 0.5 to 0.6   8/3 - Trying (need to retrain)
# * Try AutoKeras for hyper parameter tuning.
# * Try sklearn skitOptimizer
# * Try AutoGluon https://towardsdatascience.com/autogluon-deep-learning-automl-5cdb4e2388ec 
# 
# * Recalculate all the weights for ResNet50
# * Try MXNet is faster than tensorflow.
# * Add a custom layer which included the forgery decision to determine the weights (maybe use a MixMaxScaler to maximize the diff range)
# * Try a VGG16 with triplet loss mining.
# * Build a custom layers
# 
# 

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


# In[2]:


import tensorflow as tf
print(tf.__version__)


# In[3]:


imageDimensions = (224,224,3)

## Location to save the DNN model to generate Embeddings.
filename = '/notebooks/capstone/models/embeddings-res32'        ##DNN Embeddings models save location
filename2 = '/notebooks/capstone/models/siamesenetwork-res32'   ##DNN Siamesemodelsave location

## Locations to pickle the generated embeddings for training data using trained model. 
#pFile_embeddings_train = "/notebooks/capstone/train_embeddings.pickle"  ##cosine
pFile_embeddings_train = "/notebooks/capstone/train_embeddings_k.pickle"  ##kmean distance

## Locations to pickle the generated embeddings for test data using trained model. 
pFile_embeddings_tst = "/notebooks/capstone/tst_embeddings_k.pickle"

##LogisticRegression model saved.
#pFile_lrmodel = "/notebooks/capstone/train_lr_c.pickle" #cosine
pFile_lrmodel = "/notebooks/capstone/train_lr_k.pickle"  #squared distances


# <b>Utility functions</b>

# In[4]:


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


# In[5]:


def preprocess_triplets_train( anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """
    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )


# In[6]:


def preprocess_triplets(personId, anchor, positive, negative, isGenuine, anchor_path, positive_path, negative_path):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """
    print(">>> Anchor", anchor)

    return (
        personId,
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
        isGenuine,
        anchor_path, positive_path, negative_path
    )


# In[7]:


def visualize(images_dataset):
    """Visualize a few triplets from the supplied batches."""

    def showImages(ax, image):
        for i in range(3):
            ax[i].imshow(image[i])
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)

    rows = len(images_dataset)
    images = list(images_dataset.as_numpy_iterator())
    fig, axs = plt.subplots(rows,3, sharex=True, sharey=True, figsize=(30,30))
    for x in range(rows):
         anchor, positive, negative = images[x][0],images[x][1],images[x][2]
         showImages(axs[x], (anchor, positive, negative))


# <b> Load training and test data </b>

# In[8]:


basePath = "/notebooks/capstone/dataset/dataset2/sign_data"
data_train = pd.read_csv(basePath + "/train/train_clean.csv")
data_train.sort_values(by="personId", inplace=True)


# In[9]:


data_test = pd.read_csv(basePath + "/test/test_clean.csv")
data_test.sort_values(by="personId", inplace=True)


# <b>Prepare training data</b>

# In[10]:

# In this setup each forgery is checked against a combination of anchor and genuine. Did not compare genuine against genuine to reduce 
# training time. Might give better accuracy if this is implemented.

def categorizeImages(df, typeOfData):
    personIds = df["personId"].unique()
    anchor_imgs = []
    postive_imgs = []
    negative_imgs = []
    for p in personIds:
        genuine = df[(df.personId==p) & (df.Genuine==1)]
        forg = df[(df.personId==p) & (df.Genuine==0)]
        anchor_img = basePath+"/"+typeOfData+"/"+genuine.iloc[0].relPath + "/"+genuine.iloc[0].fileName  
        for g in genuine[1:].index:
            pos_img= basePath+"/"+typeOfData+"/"+genuine.loc[g].relPath + "/"+genuine.loc[g].fileName  
            for f in forg.index:
                neg_img =  basePath+"/"+typeOfData+"/"+forg.loc[f].relPath + "/"+forg.loc[f].fileName  
                anchor_imgs.append(anchor_img)
                postive_imgs.append(pos_img)
                negative_imgs.append(neg_img)
    
    return anchor_imgs,postive_imgs,negative_imgs


# In[11]:


anchor_images,positive_images,negative_images = categorizeImages(data_train, "train")


# In[12]:


anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset  = tf.data.Dataset.from_tensor_slices(positive_images)
negative_dataset  = tf.data.Dataset.from_tensor_slices(negative_images)
dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets_train)


# In[13]:


# Let's now split our dataset in train and validation.
image_count = len(anchor_dataset)
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))


# In[14]:


#visualize(train_dataset.take(5))


# In[15]:


train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(8)

val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(8)


# <b>Define Evaluation Functions</b>

# In[16]:


# Compare each positive each positive in the set and and each negative against all other positives. This is used to generate embedding
# when comparing distances. Not used to get the weights of the Siamese Model.
def categorizeTestImages(df, typeOfData):
    personIds = df["personId"].unique()
    anchor_imgs = []
    postive_imgs = []
    toCompare_imgs = []
    isGenuine = []
    personId = []
    for p in personIds:
        genuine = df[(df.personId==p) & (df.Genuine==1)]
        forg = df[(df.personId==p) & (df.Genuine==0)]
        anchor_img = basePath+"/"+typeOfData+"/"+genuine.iloc[0].relPath + "/"+genuine.iloc[0].fileName  
        for g in genuine[1:].index:
            pos_img= basePath+"/"+typeOfData+"/"+genuine.loc[g].relPath + "/"+genuine.loc[g].fileName  
            #Compare this with all forgeries
            for f in forg.index:
                toCompareImg =  basePath+"/"+typeOfData+"/"+forg.loc[f].relPath + "/"+forg.loc[f].fileName 
                personId.append(p)
                anchor_imgs.append(anchor_img)
                postive_imgs.append(pos_img)
                toCompare_imgs.append(toCompareImg)
                isGenuine.append(False)
                
            # Compare current postive with all other positives besides the anchor
            for f in genuine[1:].index:
                toCompareImg =  basePath+"/"+typeOfData+"/"+genuine.loc[f].relPath + "/"+genuine.loc[f].fileName  
                if ( pos_img != toCompareImg):
                    personId.append(p)
                    anchor_imgs.append(anchor_img)
                    postive_imgs.append(pos_img)
                    toCompare_imgs.append(toCompareImg)
                    isGenuine.append(True)    
    return personId, anchor_imgs,postive_imgs,toCompare_imgs, isGenuine


# In[17]:


#visualize(tst_dataset.take(5))


# In[18]:


def prepareEvaluationData(data_df, typeOfData):
    personId, anchor_images,positive_images,toCompare_imgs, isGenuine = categorizeTestImages(data_df, typeOfData)
    personId_dataset = tf.data.Dataset.from_tensor_slices(personId)
    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
    positive_dataset  = tf.data.Dataset.from_tensor_slices(positive_images)
    toCompare_imgs_dataset  = tf.data.Dataset.from_tensor_slices(toCompare_imgs)
    isGenuine_dataset  = tf.data.Dataset.from_tensor_slices(isGenuine)

    anchor_path_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
    positive_path_dataset  = tf.data.Dataset.from_tensor_slices(positive_images)
    toCompare_imgs_path_dataset  = tf.data.Dataset.from_tensor_slices(toCompare_imgs)

    dataset = tf.data.Dataset.zip((personId_dataset, anchor_dataset, positive_dataset, toCompare_imgs_dataset, isGenuine_dataset, anchor_path_dataset, positive_path_dataset, toCompare_imgs_path_dataset))
    ##dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(preprocess_triplets)
    
    return dataset


def getEmbeddings(row):
    personId, anchor, positive, toCcompare, isGenuine, anchor_path, positive_path, toCcompare_path = row
    return (
        personId.numpy(), 
        embedding(resnet.preprocess_input(anchor)),
        embedding(resnet.preprocess_input(positive)),
        embedding(resnet.preprocess_input(toCcompare)),
        isGenuine.numpy()
        ,anchor_path.numpy()
        ,positive_path.numpy()
        ,toCcompare_path.numpy()
    )


# In[ ]:


def getEmbeddingDataFrame(data_df, typeOfData):
    dataset = prepareEvaluationData(data_df, typeOfData)
    dataset = dataset.batch(1, drop_remainder=False)
    embedding_data = [getEmbeddings(row) for row in iter(dataset)]
    embeddings_data_df = pd.DataFrame(columns=["personId", "anchor_embedding", "positive_embedding","negative_embedding","isGenuine", "Anchor_Path", "Pos_Path", "ToComparePath"], data=embedding_data)
    return embeddings_data_df


# In[22]:

#Cosine distance was not used as squared distance gave better performance
# def compareEmbeds_row_cosine (personId, anchor_embedding, positive_embedding, negative_embedding, isGenuine):
#     cosine_similarity = metrics.CosineSimilarity()

#     positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
#     #print("Positive similarity:", positive_similarity.numpy())

#     negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
#     #print("Negative similarity", negative_similarity.numpy())
#     return (positive_similarity.numpy(), negative_similarity.numpy(), (positive_similarity.numpy() - negative_similarity.numpy()))

def compareEmbeds_row_Sqrddistance (personId, anchor_embedding, positive_embedding, negative_embedding, isGenuine):
        ap_distance = tf.reduce_sum(tf.square(anchor_embedding - positive_embedding), -1)
        an_distance = tf.reduce_sum(tf.square(anchor_embedding - negative_embedding), -1)
        return (ap_distance, an_distance, (ap_distance-an_distance))

def compareEmbeds(df):
    df2 = df.apply(lambda row: compareEmbeds_row_Sqrddistance(row.personId, row.anchor_embedding, row.positive_embedding, row.negative_embedding, row.isGenuine), axis=1,  result_type='expand')
    df2.columns = ["Pos", "Neg", "Diff"]

    df[["Pos", "Neg", "Diff"]]=df2[["Pos", "Neg", "Diff"]]
    df["personId"]=df["personId"].apply(lambda x: x[0])
    df["isGenuine"]=df["isGenuine"].apply(lambda x: x[0])
    
    return df


# <b> Prepare model for Triplet Loss Mining </b> to compute the weights to generate embeddings for a signature specimen which will be closer to the genuine anchor signature and furthest from a forgery signature<br><br>
# <b>Prepare Model Architecture</b>
# Setting up the embedding generator model

# In[ ]:

if "__main__" == __name__:
    base_cnn = resnet.ResNet50(
        weights="imagenet", input_shape=imageDimensions, include_top=False
    )

    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(dense2)

    embedding = Model(base_cnn.input, output, name="Embedding")

    trainable = False
    for layer in base_cnn.layers:
    #     if layer.name == "conv5_block1_out":     ##TODO: Why only this layer?
    #         trainable = True
        layer.trainable = trainable


    # In[ ]:


    embedding.summary()


    # <B>Setting up the Siamese Network model</B>

    # In[ ]:


    class DistanceLayer(layers.Layer):
        """
        This layer is responsible for computing the distance between the anchor
        embedding and the positive embedding, and the anchor embedding and the
        negative embedding.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def call(self, anchor, positive, negative):
            ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
            an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
            return (ap_distance, an_distance)


    anchor_input = layers.Input(name="anchor", shape=imageDimensions)
    positive_input = layers.Input(name="positive", shape=imageDimensions)
    negative_input = layers.Input(name="negative", shape=imageDimensions)

    distances = DistanceLayer()(
        embedding(resnet.preprocess_input(anchor_input)),   ##TODO : What is pre-process input do here?
        embedding(resnet.preprocess_input(positive_input)),
        embedding(resnet.preprocess_input(negative_input)),
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances  ##TODO: Not clear what Output does here?
    )


    # <b> Train Model - To Generate Embedding </b>
    # <p>We now need to implement a model with custom training loop so we can compute the triplet loss using the three embeddings produced by the Siamese network.</p>

    # In[ ]:


    class SiameseModel(Model):
        """The Siamese Network model with a custom training and testing loops.

        Computes the triplet loss using the three embeddings produced by the
        Siamese Network.

        The triplet loss is defined as:
           L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
        """

        def __init__(self, siamese_network, margin=0.5):
            super(SiameseModel, self).__init__()
            self.siamese_network = siamese_network
            self.margin = margin
            self.loss_tracker = metrics.Mean(name="loss")

        def call(self, inputs):
            return self.siamese_network(inputs)

        def train_step(self, data):
            # GradientTape is a context manager that records every operation that
            # you do inside. We are using it here to compute the loss so we can get
            # the gradients and apply them using the optimizer specified in
            # `compile()`.
            with tf.GradientTape() as tape:
                loss = self._compute_loss(data)

            # Storing the gradients of the loss function with respect to the
            # weights/parameters.
            gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

            # Applying the gradients on the model using the specified optimizer
            self.optimizer.apply_gradients(
                zip(gradients, self.siamese_network.trainable_weights)
            )

            # Let's update and return the training loss metric.
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

        def test_step(self, data):
            loss = self._compute_loss(data)

            # Let's update and return the loss metric.
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}


        def _compute_loss_sq(self, data):
            # The output of the network is a tuple containing the distances
            # between the anchor and the positive example, and the anchor and
            # the negative example.
            ap_distance, an_distance = self.siamese_network(data)

            # Computing the Triplet Loss by subtracting both distances and
            # making sure we don't get a negative value.
            loss = ap_distance - an_distance
            loss = tf.maximum(loss + self.margin, 0.0)
            return loss

    #     def _compute_loss_cos(self, data):
    #         cosine_similarity = metrics.CosineSimilarity()
    #         sitive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
    #         print("Positive similarity:", positive_similarity)
    #         negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
    #         print("Negative similarity", negative_similarity)
    #         loss = tf.math.squared_difference(positive_similarity, negative_similarity)
    #         loss = tf.maximum(loss + self.margin, 0.0)
    #         #tf.math.squared_difference
    #         return loss

        def _compute_loss(self, data):
            return self._compute_loss_sq(data)

        @property
        def metrics(self):
            # We need to list our metrics here so the `reset_states()` can be
            # called automatically.
            return [self.loss_tracker]


    # In[ ]:


    from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
    earlyStopping = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=3,
                                  verbose=1)

    early_stop=[earlyStopping]
    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.Adam(0.0000001))
    siamese_model.fit(train_dataset, epochs=40, validation_data=val_dataset, callbacks=early_stop)  #8/8 - Changes from 20 to 40 to see if improve perf

    #Tuning
    # Image dimension=448, margin=0.5 epoch 1, norestraining of weights,Adam=0.0001, loss=2.1
    # Image dimension=224, margin=0.5 epoch 1, norestraining of weights,Adam=0.0001, loss=0.05
    # Image dimension=224, margin=0.5 epoch 1, norestraining of weights,Adam=0.0001, lossFn=cos, loss=? 
    # Image dimension=224, margin=10 epoch 1, norestraining of weights,Adam=0.0001, loss=0.5
    # Image dimension=224, margin=0.5 epoch 1, norestraining of weights, Adam=0.00001 instead of 0.0001 loss=?**  0.000001 - 0.0020 - val_loss: 0.0095
    # Image dimension=112, margin=0.5 epoch 1, norestraining of weights,Adam=0.0001, loss=0.17


    # In[ ]:


    embedding.save(filename)
    siamese_network.save(filename2)


    # In[ ]:


    siamese_network = tf.keras.models.load_model(filename2)


    # <b>Generate/Compare Embedding related utility functions</b>

    # In[20]:




    # <b> Train Decision Model</b>

    # In[22]:


    ##Load model
    embedding = tf.keras.models.load_model(filename)


    # In[ ]:


    embeddings_data_df_train = getEmbeddingDataFrame(data_train, "train")
    embeddings_data_df_train.head(1)


    # In[ ]:


    with open(pFile_embeddings_train, 'wb') as file:
        pickle.dump(embeddings_data_df_train, file)


    # In[19]:


    with open(pFile_embeddings_train, 'rb') as file:
        df = pickle.load(file)
    df.shape


    # In[10]:


    # Compare embedding with the anchor
    df = compareEmbeds(df)
    df.head(3)


    # In[11]:


    df_train_forgery = df.assign(isForgery=lambda x: (0 == x.isGenuine))
    df_train_forgery.head(3)


    # In[13]:


    # Setup a logistic regression model for predictions
    #lr = LogisticRegression(random_state=0).fit(df[["personId", "Pos", "Neg"]], df["isGenuine"])
    lr = LogisticRegression(random_state=0).fit(df_train_forgery[["personId", "Pos", "Neg"]], df_train_forgery["isForgery"])


    # In[14]:


    with open(pFile_lrmodel, 'wb') as file:
        pickle.dump(lr, file)


    # In[26]:


    with open(pFile_lrmodel, 'rb') as file:
        lr = pickle.load(file)


    # In[16]:


    y_train_predict = lr.predict(df_train_forgery[["personId", "Pos", "Neg"]])  #neg = tocompare


    # In[17]:


    # fpr, tpr, thresholds = metrics.roc_curve(df["isGenuine"], y_train_predict, pos_label=2)
    # metrics.auc(fpr, tpr)
    m_train = metrics.AUC()
    m_train.update_state(df_train_forgery["isForgery"], y_train_predict)
    m_train.result().numpy()


    # In[18]:


    #metrics.confusion_matrix(df["isGenuine"], y_train_predict)
    tf.math.confusion_matrix(df_train_forgery["isForgery"], y_train_predict)


    # <b>Testing</b>

    # Inspecting what the network has learned
    # At this point, we can check how the network learned to separate the embeddings depending on whether they belong to similar images.
    # 
    # We can use cosine similarity to measure the similarity between embeddings.
    # 
    # Let's pick a sample from the dataset to check the similarity between the embeddings generated for each image.

    # Finally, we can compute the cosine similarity between the anchor and positive images and compare it with the similarity between the anchor and the negative images.
    # 
    # We should expect the similarity between the anchor and positive images to be larger than the similarity between the anchor and the negative images.

    # In[ ]:


    ##Load model
    embedding = tf.keras.models.load_model(filename)


    # In[ ]:


    embeddings_data_df_tst = getEmbeddingDataFrame(data_test, "test")
    embeddings_data_df_tst.head(1)


    # In[ ]:


    with open(pFile_embeddings_tst, 'wb') as file:
        pickle.dump(embeddings_data_df_tst, file)


    # In[20]:


    with open(pFile_embeddings_tst, 'rb') as file:
        df_tst = pickle.load(file)
    df_tst.shape


    # In[23]:


    df_tst = compareEmbeds(df_tst)
    df_tst.head(3)


    # In[25]:


    df_test_forgery = df_tst.assign(isForgery=lambda x: (0 == x.isGenuine))
    df_test_forgery.head(5)


    # In[27]:


    y_tst_predict = lr.predict(df_test_forgery[["personId", "Pos", "Neg"]])  #neg = tocompare


    # In[28]:


    # from sklearn import metrics
    # fpr_tst, tpr_tst, thresholds_tst = metrics.roc_curve(df_tst["isGenuine"], y_tst_predict, pos_label=2)
    # metrics.auc(fpr_tst, tpr_tst)
    m_tst = metrics.AUC()
    m_tst.update_state(df_test_forgery["isForgery"], y_tst_predict)
    m_tst.result().numpy()


    # In[29]:


    #metrics.confusion_matrix(df_tst["isGenuine"], y_tst_predict)
    tf.math.confusion_matrix(df_test_forgery["isForgery"], y_tst_predict)


    # In[28]:


    m_tst_recall = metrics.Recall()
    m_tst_recall.update_state(df_test_forgery["isForgery"], y_tst_predict)
    m_tst_recall.result().numpy()


    # In[29]:


    m_tst_precision = metrics.Precision()
    m_tst_precision.update_state(df_test_forgery["isForgery"], y_tst_predict)
    m_tst_precision.result().numpy()

