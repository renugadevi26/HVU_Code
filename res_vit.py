import cv2
import os
import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras_vit.vit import ViT_B16
import keras_vit.layers as layers
layers.tf = tf


cl=os.listdir("/nfsshare/knr/knrtrain/renu/figshare/figshare-dataset/")
lb=[]
imgs=[]
for i in cl:
    mn=f'/nfsshare/knr/knrtrain/renu/figshare/figshare-dataset/{i}'
    idx=cl.index(i)
    for j in os.listdir(mn):
        img=cv2.imread(os.path.join(mn,j))
        r=cv2.resize(img,(256,256))
        imgs.append(r)
        lb.append(idx)
import numpy as np
imgs=np.array(imgs)
lb=np.array(lb)


#one-hot
lb=tf.keras.utils.to_categorical(lb,num_classes=3)


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
# Define input shape
input_shape = (256, 256, 3)

# Load the InceptionV3 model with pretrained weights set to None


def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p
def build_unet(input_shape):
    # Load ResNet-50 with pre-trained weights
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    layer_to_keep = base_model.layers[-36].output

    vit = ViT_B16(image_size=256,pre_trained=False,include_mlp_head=256)

    # Create a model that includes both the ViT and ResNet models
    combined_input = base_model.input
    vit_output = vit.layers[-2]
    vit_output=vit_output(combined_input)
    # Global Average Pooling
    
    vit_output=tf.keras.layers.Reshape((16,16,3))(vit_output)
    
    # U-Net architecture
    inputs = base_model.input  # Use the input from the loaded ResNet-50 model
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)
    b1 = Concatenate()([b1, layer_to_keep,vit_output])
    
    d1= Flatten()(b1)
    d2= Dense(128,activation='relu')(d1)
    outputs= Dense(3,activation='softmax')(d2)
    
    

    model = Model(inputs, outputs, name="UNET")
    return model


unet_model = build_unet((256, 256, 3))


# Computing Precision 
def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    
# Computing Sensitivity      
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
    
    
def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.math.round(y_pred), tf.float32)

    # True Positives
    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    # False Positives
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    # False Negatives
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

    # Calculate precision and recall
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    # Average F1 score across all classes
    f1 = tf.reduce_mean(f1)

    return f1

# Compile the model
unet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',precision,sensitivity,specificity,tf.keras.metrics.Recall(),f1_score])
callbacks = [
    ModelCheckpoint('/nfsshare/knr/knrtrain/renu/figshare/vit_res/vit_res.weights.h5', verbose=1, save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger('/nfsshare/knr/knrtrain/renu/figshare/vit_res/vit_res.csv')
]
unet_model.fit(imgs,lb,epochs=100,validation_split=0.2,batch_size=32,callbacks=callbacks)