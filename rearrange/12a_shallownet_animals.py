from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.cnn import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='Path to input dataset')
args = vars(ap.parse_args())

# Grab the list of images
print('[INFO]: Loading images....')
image_paths = list(paths.list_images(args['dataset']))

# Initialize the preprocessors
sp = SimplePreprocessor(32, 32)
itap = ImageToArrayPreprocessor()

# Load the dataset and scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, itap])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype('float') / 255.0

# Split the data into training data (75%) and testing data (25%)
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Convert the labels from integers to vectors
train_y = LabelBinarizer().fit_transform(train_y)
test_y = LabelBinarizer().fit_transform(test_y)

# Initialize the optimizer and model
print('[INFO]: Compiling model....')
optimizer = SGD(learning_rate=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the network
print('[INFO]: Training the network....')
num_epochs=100
H = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=32, epochs=num_epochs, verbose=1)

# Test the network
print('[INFO]: Evaluating the network....')
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=['cat', 'dog', 'panda']))

# Plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, num_epochs), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, num_epochs), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, num_epochs), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, num_epochs), H.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
