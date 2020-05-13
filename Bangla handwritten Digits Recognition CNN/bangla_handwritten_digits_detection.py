from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

#our bangla digit(train) image is in png format...
# means it is eligible to be rgba format. the a = alpha is to be removed before further processing
def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im

# returns like mnist dataset from keras
def load_data():
    X_train = []
    y_train = []

    X_test = []
    y_test = []


    # training
    # train one
    fileNameList = os.listdir(r"F:\python codes\interview_codes\bangla_digits\one")
    os.chdir(r"F:\python codes\interview_codes\bangla_digits\one")

    for fileName in fileNameList:
        image = Image.open(fileName,"r")
        image = remove_transparency(image)
        image = image.convert('L')
        image_array = np.array(image.getdata())
        image_array = 255 - image_array
        image_array = image_array.reshape(40, 40)
        # plt.imshow(image_array,cmap='gray')
        # plt.show()
        X_train.append(image_array.tolist())
        #print(X_train)
        y_train.append(1)

    # train two
    fileNameList = os.listdir(r"F:\python codes\interview_codes\bangla_digits\two")
    os.chdir(r"F:\python codes\interview_codes\bangla_digits\two")

    for fileName in fileNameList:
        image = Image.open(fileName, "r")
        image = remove_transparency(image)
        image = image.convert('L')
        image_array = np.array(image.getdata())
        image_array = 255 - image_array
        image_array = image_array.reshape(40, 40)
        # plt.imshow(image_array, cmap='gray')
        # plt.show()
        X_train.append(image_array.tolist())
        #print(X_train)
        y_train.append(2)

    # train three
    fileNameList = os.listdir(r"F:\python codes\interview_codes\bangla_digits\three")
    os.chdir(r"F:\python codes\interview_codes\bangla_digits\three")

    for fileName in fileNameList:
        image = Image.open(fileName, "r")
        image = remove_transparency(image)
        image = image.convert('L')
        image_array = np.array(image.getdata())
        image_array = 255 - image_array
        image_array = image_array.reshape(40, 40)
        # plt.imshow(image_array, cmap='gray')
        # plt.show()
        X_train.append(image_array.tolist())
        #print(X_train)
        y_train.append(3)

    # train four
    fileNameList = os.listdir(r"F:\python codes\interview_codes\bangla_digits\four")
    os.chdir(r"F:\python codes\interview_codes\bangla_digits\four")

    for fileName in fileNameList:
        image = Image.open(fileName, "r")
        image = remove_transparency(image)
        image = image.convert('L')
        image_array = np.array(image.getdata())
        image_array = 255 - image_array
        image_array = image_array.reshape(40, 40)
        # plt.imshow(image_array, cmap='gray')
        # plt.show()
        X_train.append(image_array.tolist())
        #print(X_train)
        y_train.append(4)



    # now testing
    # test one
    fileNameList = os.listdir(r"F:\python codes\interview_codes\bangla_digits\one_test")
    os.chdir(r"F:\python codes\interview_codes\bangla_digits\one_test")

    for fileName in fileNameList:
        image = Image.open(fileName, "r")
        image = remove_transparency(image)
        image = image.convert('L')
        image_array = np.array(image.getdata())
        image_array = 255 - image_array
        image_array = image_array.reshape(40, 40)
        # plt.imshow(image_array, cmap='gray')
        # plt.show()
        X_test.append(image_array.tolist())
        #print(X_train)
        y_test.append(1)

    # test two
    fileNameList = os.listdir(r"F:\python codes\interview_codes\bangla_digits\two_test")
    os.chdir(r"F:\python codes\interview_codes\bangla_digits\two_test")

    for fileName in fileNameList:
        image = Image.open(fileName, "r")
        image = remove_transparency(image)
        image = image.convert('L')
        image_array = np.array(image.getdata())
        image_array = 255 - image_array
        image_array = image_array.reshape(40, 40)
        # plt.imshow(image_array, cmap='gray')
        # plt.show()
        X_test.append(image_array.tolist())
        #print(X_train)
        y_test.append(2)

    # test three
    fileNameList = os.listdir(r"F:\python codes\interview_codes\bangla_digits\three_test")
    os.chdir(r"F:\python codes\interview_codes\bangla_digits\three_test")

    for fileName in fileNameList:
        image = Image.open(fileName, "r")
        image = remove_transparency(image)
        image = image.convert('L')
        image_array = np.array(image.getdata())
        image_array = 255 - image_array
        image_array = image_array.reshape(40, 40)
        # plt.imshow(image_array, cmap='gray')
        # plt.show()
        X_test.append(image_array.tolist())
        #print(X_train)
        y_test.append(3)

    # test four
    fileNameList = os.listdir(r"F:\python codes\interview_codes\bangla_digits\four_test")
    os.chdir(r"F:\python codes\interview_codes\bangla_digits\four_test")

    for fileName in fileNameList:
        image = Image.open(fileName, "r")
        image = remove_transparency(image)
        image = image.convert('L')
        image_array = np.array(image.getdata())
        image_array = 255 - image_array
        image_array = image_array.reshape(40, 40)
        # plt.imshow(image_array, cmap='gray')
        # plt.show()
        X_test.append(image_array.tolist())
        #print(X_train)
        y_test.append(4)


    return (X_train, y_train),(X_test, y_test)



(X_train, y_train),(X_test, y_test) = load_data()

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
# (66, 40, 40) (66,) (20, 40, 40) (20,)
# 66 train data,20 test data


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)


# Data normalization
# We then normalize the data dimensions so that they are of approximately the same scale.

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# 1-hot encoding to target data
# y data(train+test) must be 2d vectors for using tensorflow
from keras.utils import to_categorical
print("before to_categorical: ",y_train)
# 11112222233334444 so the num_class will be 5 in support 4 as index
num_classes = 5
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print("after to_categorical: ",y_train)


# define the model
# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()

# model takes input_shape as 3d image(RGB)
input_shape = (40, 40, 1)
model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(5,activation=tf.nn.softmax))

# We use model.compile() to configure the learning process
# before training the model. This is where you define the type of loss function,
# optimizer and the metrics evaluated by the model during training and testing.

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# We will train the model with a batch_size of 64 and 10 epochs.
print("model summary: ")
print(model.summary())

model.fit(X_train,
         y_train,epochs=1)


print(model.predict(X_test,verbose=0).round(decimals=2))


# test specific one image
# this is 1
# show this image
print(X_train.shape)
X_train = X_train.reshape(66,40,40)
plt.imshow(X_train[5])
plt.show()


#predict this test data
X_train = X_train.reshape(66,40,40,1)
y_pred = model.predict([[X_train[5]]],verbose=0)
print("prediction: ",y_pred)
# [[0.09428898 0.46514738 0.15075094 0.08279515 0.20701751]]
# so, it could predict accurate result(1=prabability=0.45=max probability)


# test specific one image
# this is 4
# show this image
print(X_train.shape)
X_train = X_train.reshape(66,40,40)
plt.imshow(X_train[60])
plt.show()


#predict this test data
X_train = X_train.reshape(66,40,40,1)
y_pred = model.predict([[X_train[60]]],verbose=0)
print("prediction: ",y_pred)
# [[0.05981155 0.11513592 0.1275378  0.12577239 0.57174236]]
# so, it could predict accurate result(4=prabability=0.57=max probability)


