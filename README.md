# coral-classifier
Classifying corals with Tensorflow

![Brain Coral](https://github.com/churst12/coral-classifier/blob/master/coral.jpg)

This repository uses transfer learning on the VGG19 model for the purpose of classifying coral species.



##File Functionality

For all scripts, the config settings are at the top of the file.
### app3.py

The file used to train the model. This file on first setup downloads VGG19, which should take a while. After that, it will access the cached VGG Model.
This script requires you to have separate folders setup for your species, which can be done with script.py

You should probably run this with a beefy computer if you have 2000+ photos.

### app3load.py

This file is used to test the trained model. It requires you to have validation data set up in folders. The script will then match the model accuracies with the validation data and plot them accordingly

###script.py

This file is used to download the observations from a csv retrieved from iNaturalist. 
The input csv must be from iNaturalist since it reads and downloads based on the iNaturalist csv format.
