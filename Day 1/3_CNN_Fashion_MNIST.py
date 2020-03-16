from keras import models
from keras import layers
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras import regularizers

# Load the MNIST data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print('train_images shape:', train_images.shape)
print('test_images shape:', test_images.shape)
print('train_labels shape:', train_labels.shape)

#Display a sample
# extract_image = 10
# digit = train_images[extract_image]
# label = train_labels[extract_image]
# print("Label =",label)
# import matplotlib.pyplot as plt
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

# Process the data for the usage of ANN
train_images = train_images.reshape((60000, 28, 28, 1)) # Flatten the image
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28,28,1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Create the network
network = models.Sequential()
# Convert to convolution layer
network.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
#network.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
network.add(layers.Flatten())
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history = network.fit(train_images, train_labels, validation_split=0.2, epochs=5, batch_size=128, verbose=2)

test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=2)
print('test_acc:', test_acc)


# list all data in history
print(history.history.keys())

# Plot the Learning curve
import matplotlib.pyplot as plt
history_dict = history.history
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.plot(epochs, acc_values, 'ro', label='Training acc')
plt.plot(epochs, val_acc_values, 'r', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Acc/Loss')
plt.legend()
plt.show()


