from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

img_width, img_height = 256, 256
train_data_dir = "photos/test1"
validation_data_dir = "photos/test1val"
nb_train_samples = 185
nb_validation_samples = 56
batch_size = 16
epochs = 50

def create_model():
	model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

	# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
	for layer in model.layers[:5]:
	    layer.trainable = False

	#Adding custom Layers 
	x = model.output
	x = Flatten()(x)
	x = Dense(1024, activation="relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(1024, activation="relu")(x)
	predictions = Dense(7, activation="softmax")(x)

	# creating the final model 
	model_final = Model(input = model.input, output = predictions)

	return model_final

def load_weights(path):
	model = create_model()
	model.load_weights(path)
	return model

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(256, 256))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor
def imgaccuracy(imgpath):
	model = load_weights('vgg16_1.h5')
	newimg = load_image(imgpath)
	pred = model.predict(newimg)
	return pred

def print_results(directorystr):
	dsflag = True
	
	directory = os.fsencode(directorystr)
	for folder in os.listdir(directory):

		print("folder" + str(folder.decode("utf-8")))
		directory2 = os.fsencode(directorystr+'/'+str(folder.decode("utf-8")))
		if dsflag: dsflag = False
		else:
			for file in os.listdir(directory2):
				if file.decode("utf-8") == ".DS_Store": continue
				else:
					prediction = imgaccuracy(directorystr+"/"+folder.decode("utf-8")+"/"+file.decode("utf-8"))
					print(folder.decode("utf-8") + file.decode("utf-8"))
					print(prediction)
				




print_results('/Users/collin.hurst/Documents/coral/photos/test1val')





