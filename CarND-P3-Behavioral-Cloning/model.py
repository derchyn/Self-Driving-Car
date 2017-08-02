import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

Input_Shape = (160, 320, 3)
Batch_Size = 32
Epochs =3

samples=[]
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
del(samples[0])

def generator(samples, batch_size=Batch_Size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        aug_imgs = []
        aug_measures = []
        for offset in range(0, num_samples, batch_size):
            batch_samples=samples[offset: offset+batch_size]
            for line in batch_samples:
                line_path = line[0]
                steering = float(line[3])
                # Create adjusted steering measurements for the side camera images
                correction = float(0.25)
                camera = np.random.choice(['center', 'left', 'right'])
                if(camera == 'left'):
                    line_path = line[1]
                    steering += correction
                elif(camera == 'right'):
                    line_path = line[2]
                    steering -= correction

                filename = line_path.split('/')[-1]
                img_path = 'data/IMG/' + filename
                image = cv2.imread(img_path)

                # decide whether to horizontally flip the image
                flip_prob = np.random.random()
                if(flip_prob > 0.5):
                    # flip the image and reverse the steering angle
                    steering = -1.0 * steering
                    image = cv2.flip(image, 1)
                
                aug_imgs.append(image)
                aug_measures.append(steering)

            train_X = np.array(aug_imgs)
            train_y = np.array(aug_measures)

            yield (train_X, train_y)

training_samples, validation_samples = train_test_split(samples, test_size=0.2)
            
# Compile and train the model using the generator function
training_generator = generator(training_samples, batch_size=Batch_Size)
validation_generator = generator(validation_samples, batch_size=Batch_Size)

# End to End Learning for Self-Driving Cars, https://arxiv.org/abs/1604.07316
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=Input_Shape))
model.add(Cropping2D(cropping=((65,20), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(training_generator, 
                    samples_per_epoch = len(training_samples), 
                    validation_data = validation_generator,
                    nb_val_samples = len(validation_samples), 
                    nb_epoch=Epochs, 
                    verbose=1)

model.summary()

# save the model 
model.save('model.h5') 
