import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf #adanya tensorflow memungkinkan kita untuk tidak usah mengunduh file csv yang berisi dataset. Kita sudah punya dataset yang butuh di training

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# model = tf.keras.models.Sequential() #model umum dari neural network
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) #28, 28 adalah ukuran dari angka yang kita set di paint
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax')) 

# model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# model.fit(x_train, y_train, epochs=3) #epoch adalah hidden layer yang dapat meningkatkan akurasi prediksi

# model.save('Handwrite regocnition')

model = tf.keras.models.load_model('Handwrite regocnition')

image_number = 1
while os.path.isfile(f"Data/angka{image_number}.png"):
    try:
        img = cv2.imread(f"Data/angka{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"Ini adalah angka {np.argmax(prediction)}")
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print("error")
    finally:
        image_number += 1
