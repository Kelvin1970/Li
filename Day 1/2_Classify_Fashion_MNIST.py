from keras import models
from keras import layers
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

# Load the MNIST data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print('train_images shape:', train_images.shape)
print('test_images shape:', test_images.shape)
print('train_labels shape:', train_labels.shape)


#Display a sample
# extract_digit = 10
# digit = train_images[extract_digit]
# label = train_labels[extract_digit]
# print("Label =",label)
# import matplotlib.pyplot as plt
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()


# Process the data for the usage of ANN
train_images = train_images.reshape((60000, 28 * 28)) # Flatten the image
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Create the network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history = network.fit(train_images, train_labels, epochs=5, batch_size=128)

# list all data in history
print(history.history.keys())

# Plot the Learning curve
import matplotlib.pyplot as plt
history_dict = history.history
acc_values = history_dict['acc']
loss_values = history_dict['loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, acc_values, 'bo', label='Training Accuracy')
plt.plot(epochs, loss_values, 'b', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Acc/Lost')
plt.legend()
plt.show()

test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=2)
print('test_acc:', test_acc)
print("-----------------------------------")

