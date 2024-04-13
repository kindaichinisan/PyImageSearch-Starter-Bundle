from pyimagesearch.nn.cnn import LeNet
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

# Grab the MNIST dataset
print('[INFO]: Accessing MNIST....')
dataset = datasets.fetch_openml('mnist_784')
data = dataset.data
data = data.values #convert DataFrame to np array


# 'channels_first' ordering
if K.image_data_format() == "channels_first":
    # Reshape the design matrix such that the matrix is: num_samples x depth x rows x columns
    data = data.reshape(data.shape[0], 1, 28, 28)
# 'channels_last' ordering
else:
    # Reshape the design matrix such that the matrix is: num_samples x rows x columns x depth
    data = data.reshape(data.shape[0], 28, 28, 1)

# Scale the input data to the range [0, 1] and perform a train/test split
(train_x, test_x, train_y, test_y) = train_test_split(data / 255.0, dataset.target.astype("int"), test_size=0.25, random_state=42)

# Convert the labels from integers to vectors
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# Initialize the optimizer and model
print("[INFO]: Compiling model....")
optimizer = SGD(learning_rate=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the network
print("[INFO]: Training....")
num_epochs=20
H = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=128, epochs=num_epochs, verbose=1)

# Evaluate the network
print("[INFO]: Evaluating....")
predictions = model.predict(test_x, batch_size=128)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, num_epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, num_epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, num_epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, num_epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
