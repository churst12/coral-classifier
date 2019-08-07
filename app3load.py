from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import os
#https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8
img_width, img_height = 256, 256
validation_datapath = "/Users/collin.hurst/Documents/coral/photos/test2val"
classes = ('Agaricia agaricites', 'Agaricia humilis', 'Agaraicia lamarcki', 'Colpophyllia natans', 'Eusmilia fastigiata', 'Madracis aurentenra', 'Madracis decactis', 'Madracis pharensis', 'Madracis senaria', 'Orbicella annularis', 'Orbicella faveolata', 'Orbicella franksi')
modelpath = '8_8_vgg.h5'
plotpath = 'results_8_7'

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
	predictions = Dense(len(classes), activation="softmax")(x)

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
	model = load_weights(modelpath)
	newimg = load_image(imgpath)
	pred = model.predict(newimg)
	return pred

def print_results(directorystr):
	
	directory = os.fsencode(directorystr)
	for folder in os.listdir(directory):
		species = str(folder.decode("utf-8"))
		print("species: " + species)
		directory2 = os.fsencode(directorystr+'/'+ species)
		if folder.decode("utf-8") == ".DS_Store": continue
		else:
			imgcounter = 0
			confidencelist = [0]*len(classes)
			
			for file in os.listdir(directory2):
				if file.decode("utf-8") == ".DS_Store": continue
				else:
					prediction = imgaccuracy(directorystr+"/"+ species +"/"+file.decode("utf-8"))
					for i in range(len(prediction)):
						confidencelist[i] += prediction[i]

					print(species + file.decode("utf-8"))
					print(prediction)
					imgcounter += 1
			for i in range(len(confidencelist)):
				confidencelist[i] = confidencelist[i]/imgcounter
			confidencelist = confidencelist[0].tolist()
			plot(species, confidencelist)
				
def plot(name, confidence):
	y_pos = np.arange(len(classes))

	print(str(y_pos))
	print("CONFIDENCE: "+ str(confidence))
	plt.ylim(top=1)
	plt.bar(y_pos, confidence, align='center', alpha=0.5)
	plt.xticks(y_pos, classes, rotation=30, ha="right")
	plt.margins(.2)

	plt.ylabel('confidence')
	plt.title(name)

	if not os.path.exists(plotpath):
		os.makedirs(plotpath)

	plt.savefig(plotpath + '/'+name, bbox_inches='tight')
	plt.close()



print_results(validation_datapath)





