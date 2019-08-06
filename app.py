from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub


from tensorflow.keras import layers

IMAGE_RES = 299
URL = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/3"
TRAIN_DIR = "/photos/testing/train"
BATCH_SIZE = 8
class_list = ['Acropora', 'Pavona', 'Turbastrea', 'Turbinaria']


def retrieve_model():
	from keras.applications.inception_v3 import InceptionV3

	base_model = InceptionV3(weights='imagenet', 
                      include_top=False, 
                      input_shape=(IMAGE_RES, IMAGE_RES, 3))


def image_augment():
	from keras.preprocessing.image import ImageDataGenerator
	train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=90,
      horizontal_flip=True,
      vertical_flip=True
    )

	train_generator = train_datagen.flow_from_directory(TRAIN_DIR, 
        target_size=(IMAGE_RES, IMAGE_RES), 
        batch_size=BATCH_SIZE)

def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x) 
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model
def finetune():
	FC_LAYERS = [1024, 1024]
	dropout = 0.5

	finetune_model = build_finetune_model(base_model, 
	                                      dropout=dropout, 
	                                      fc_layers=FC_LAYERS, 
	                                      num_classes=len(class_list))

def train():
	from keras.optimizers import SGD, Adam

	NUM_EPOCHS = 10
	BATCH_SIZE = 8
	num_train_images = 10000

	adam = Adam(lr=0.00001)
	finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

	filepath="./checkpoints/" + "ResNet50" + "_model_weights.h5"
	checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
	callbacks_list = [checkpoint]

	history = finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8, 
	                                       steps_per_epoch=num_train_images // BATCH_SIZE, 
	                                       shuffle=True, callbacks=callbacks_list)


	plot_training(history)

retrieve_model()






