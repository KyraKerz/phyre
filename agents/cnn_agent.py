import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, LeakyReLU, MaxPooling2D
import numpy as np
import animations
from keras_video import VideoFrameGenerator
from matplotlib import pyplot as plt
from keras.layers.merge import Concatenate


random.seed(0)

images = np.load('ImagesLog.npy', allow_pickle=True)
evaluation = np.load('EvaluationsLog.npy', allow_pickle=True)
actions = np.load('ActionLog.npy', allow_pickle=True)

index = [i for i, e in enumerate(evaluation) if e == 1]
print(index)
actions_artificial = actions[0:80]
actions_artificial = np.insert(actions_artificial,1,  actions[875], axis=0)
actions_artificial = np.insert(actions_artificial,10,  actions[1280], axis=0)
actions_artificial = np.insert(actions_artificial,22,  actions[875], axis=0)
actions_artificial = np.insert(actions_artificial,56,  actions[1459], axis=0)
actions_artificial = np.insert(actions_artificial,77,  actions[1515], axis=0)
actions_artificial = np.insert(actions_artificial,15,  actions[1931], axis=0)
actions_artificial = np.insert(actions_artificial,37,  actions[1635], axis=0)
actions_artificial = np.insert(actions_artificial,63,  actions[1280], axis=0)
actions_artificial = np.insert(actions_artificial,43,  actions[1635], axis=0)
actions_artificial = np.insert(actions_artificial,14,  actions[1931], axis=0)
actions_artificial = np.insert(actions_artificial,55,  actions[1459], axis=0)
actions_artificial = np.insert(actions_artificial,16,  actions[1515], axis=0)
actions_artificial = np.insert(actions_artificial,13,  actions[1635], axis=0)
actions_artificial = np.insert(actions_artificial,29,  actions[875], axis=0)
actions_artificial = np.insert(actions_artificial,85,  actions[1280], axis=0)



images_artificial = images[0:80]
images_artificial = np.insert(images_artificial,1,  images[875], axis=0)
images_artificial = np.insert(images_artificial,10,  images[1280], axis=0)
images_artificial = np.insert(images_artificial,22,  images[875], axis=0)
images_artificial = np.insert(images_artificial,56,  images[1459], axis=0)
images_artificial = np.insert(images_artificial,77,  images[1515], axis=0)
images_artificial = np.insert(images_artificial,15,  images[1931], axis=0)
images_artificial = np.insert(images_artificial,37,  images[1635], axis=0)
images_artificial = np.insert(images_artificial,63,  images[1280], axis=0)
images_artificial = np.insert(images_artificial,43,  images[1635], axis=0)
images_artificial = np.insert(images_artificial,14,  images[1931], axis=0)
images_artificial = np.insert(images_artificial,55,  images[1459], axis=0)
images_artificial = np.insert(images_artificial,16,  images[1515], axis=0)
images_artificial = np.insert(images_artificial,13,  images[1635], axis=0)
images_artificial = np.insert(images_artificial,29,  images[875], axis=0)
images_artificial = np.insert(images_artificial,85,  images[1280], axis=0)

eval_artificial = evaluation[0:80]
eval_artificial = np.insert(eval_artificial, 1,  evaluation[875], axis=0)
eval_artificial = np.insert(eval_artificial,10,  evaluation[1280], axis=0)
eval_artificial = np.insert(eval_artificial,22,  evaluation[875], axis=0)
eval_artificial = np.insert(eval_artificial,56,  evaluation[1459], axis=0)
eval_artificial = np.insert(eval_artificial,77,  evaluation[1515], axis=0)
eval_artificial = np.insert(eval_artificial,15,  evaluation[1931], axis=0)
eval_artificial = np.insert(eval_artificial,37,  evaluation[1635], axis=0)
eval_artificial = np.insert(eval_artificial,63,  evaluation[1280], axis=0)
eval_artificial = np.insert(eval_artificial,43,  evaluation[1635], axis=0)
eval_artificial = np.insert(eval_artificial,14,  evaluation[1931], axis=0)
eval_artificial = np.insert(eval_artificial,55,  evaluation[1459], axis=0)
eval_artificial = np.insert(eval_artificial,49,  evaluation[1515], axis=0)
eval_artificial = np.insert(eval_artificial,13,  evaluation[1635], axis=0)
eval_artificial = np.insert(eval_artificial,29,  evaluation[875], axis=0)
eval_artificial = np.insert(eval_artificial,56,  evaluation[1280], axis=0)

print(images_artificial.shape)
images_artificial = images_artificial.reshape(images_artificial.shape[0] * 17, 256, 256, 1)

imagesnew = images_artificial[0::17]

length =80
end = 95
eval_artificial = eval_artificial.reshape(95,)
eval_artificial = [0 if x==-1 else 1 for x in eval_artificial]
#print( evaluation.shape)
images_train = imagesnew[0:length]
actions_train = actions_artificial[0:length]
evaluation_train_values = eval_artificial[0:length]
evaluation_train = keras.utils.to_categorical(evaluation_train_values,num_classes= 2, dtype='int64')
images_validation = imagesnew[length:end]
actions_validation = actions_artificial[length:end]
evaluation_validation = keras.utils.to_categorical(eval_artificial[length:end], num_classes=2, dtype=int) #images[length:end]

input_img = keras.Input(shape=(256,256,1))
vector_input = keras.Input((3,))

x = keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same')(input_img)
x = keras.layers.MaxPooling2D((2, 2),padding='same')(x)
x = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same')(x)
x = keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same')(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same')(x)

flatten = keras.layers.Flatten()(x)# append three no: x,z radius
#concat_layer= flatten#Concatenate()([vector_input, flatten])
dense1 = keras.layers.Dense(128, activation='relu')(flatten)
dense2 = keras.layers.Dense(2, activation='softmax')(dense1)

model = keras.Model(inputs= input_img, outputs = dense2)
#model = keras.Model(inputs= [input_img, vector_input], outputs = dense2)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(images_train/7, evaluation_train, batch_size=2,epochs=30
                    ,verbose=1,validation_data=(images_validation, evaluation_validation))
#history = model.fit([images_train, actions_train], evaluation_train, batch_size=2,epochs=30,verbose=1,validation_data=([images_validation, actions_validation], evaluation_validation))
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

imagesso = np.load('ImagesPos.npy', allow_pickle=True)
animations.animateSimulatedTask(imagesso[1145])
evaluationi = np.load('EvaluationsPos.npy', allow_pickle=True)
index = [i for i, e in enumerate(evaluationi) if e == 1]
imagesso = imagesso.reshape(imagesso.shape[0] * 17, 256, 256, 1)

imagesnewi = imagesso[0::17]
print(index)
Xinew = imagesnewi[890:891]

Xnew = imagesnew[1:2]
print(Xinew.shape)
ynew = model.predict(Xinew)
# show the inputs and predicted outputs
print("X=%s, Predicted=%s" % (Xnew, ynew))