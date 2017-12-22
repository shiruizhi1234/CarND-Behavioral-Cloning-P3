import csv
import cv2
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout, BatchNormalization

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
images = []
measurements = []
correction = 0.2
for line in lines:
    steering_center = float(line[3])
    for i in range(2):
        source_path = line[i]
        if i == 0:
            measurement = steering_center
        elif i == 1:
            measurement = steering_center + correction
        else:
            measurement = steering_center - correction
        file_name = source_path.split('/')[-1]
        current_path = './data/IMG/' + file_name
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        
        measurements.append(measurement)
        ###flip image
        image_flipped = np.fliplr(image)
        measurement_flipped = -measurement
        images.append(image_flipped)
        measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)   

model = Sequential()
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape = (160, 320, 3)))
model.add(Lambda(lambda x:x / 255.0 - 0.5))

model.add(Conv2D(24, (5, 5), strides=(2,2), activation = 'relu', padding='VALID'))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation = 'relu', padding='VALID'))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation = 'relu', padding='VALID'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), strides=(1,1), activation = 'relu', padding='VALID'))
model.add(Conv2D(64, (3, 3), strides=(1,1), activation = 'relu', padding='VALID'))
model.add(Conv2D(500, (1, 1), strides=(1,1), activation = 'relu', padding='VALID'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Dropout(0.6))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss='mse')
history_object = model.fit(X_train, y_train, shuffle=True, validation_split=0.2, nb_epoch=10)

model.save('model.h5')