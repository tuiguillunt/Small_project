import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam

train_dir = './train/'
test_dir = './test/'

# Define the list of possible characters
char_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Defining variables
epochs_nb = 10
conv_size_1 = 64
conv_size_2 = 64
dense_size = 256
batch_size = 32
validation_split = 0.2
dropout_1 = 0.05
dropout_2 = 0.05

# Preparing the training data
X_train = []
y_train = []

for img_name in os.listdir(test_dir):
    img = cv2.imread(os.path.join(test_dir, img_name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            letter_img = gray[y:y+h, x:x+w]
            letter_img = cv2.resize(letter_img, (20, 20))
            X_train.append(letter_img.reshape((20, 20, 1)))
            y_train.append(img_name.split('.')[0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Convert the labels into one-hot encoded vectors to accept letters as class
le = LabelEncoder()
le.fit(char_list)
unique_chars = list(le.classes_)
y_train_encoded = le.transform(y_train)
y_train_categorical = np_utils.to_categorical(y_train_encoded, len(unique_chars))

# Preparing the testing data
X_test = []
y_test = []

# Get a list of all the captcha images and store them in a list
for img_name in os.listdir(test_dir):
    img = cv2.imread(os.path.join(test_dir, img_name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            letter_img = gray[y:y+h, x:x+w]
            letter_img = cv2.resize(letter_img, (20, 20))
            X_test.append(letter_img.reshape((20, 20, 1)))
            y_test.append(img_name.split('.')[0])

X_test = np.array(X_test)
y_test = np.array(y_test)

# Convert the labels into one-hot encoded vectors to accept letters as class
y_test_encoded = le.transform(y_test)
y_test_categorical = np_utils.to_categorical(y_test_encoded, len(unique_chars))


# Creation of the CNN model
model = Sequential()

model.add(Conv2D(conv_size_1, kernel_size=(3, 3), activation='relu', input_shape=(20, 20, 1)))
model.add(Conv2D(conv_size_2, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout_1))

model.add(Flatten())
model.add(Dense(dense_size, activation='relu'))
model.add(Dropout(dropout_2))
model.add(Dense(len(unique_chars), activation='softmax'))

adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(X_train, y_train_categorical, epochs=epochs_nb, batch_size=batch_size, validation_split=validation_split)

test_loss, test_acc = model.evaluate(X_test, y_test_categorical)
print('Test accuracy:', test_acc)

print(history.history['accuracy'][-1])
