import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
# export CUDA_VISIBLE_DEVICES=0


def load_images_and_labels(data_dir):
    images = []
    labels = []
    
    # Loop over every folder in the directory
    for class_folder in os.listdir(data_dir):
        # Loop over every image in the class folder
        for image_name in os.listdir(os.path.join(data_dir, class_folder)):
            # Open the image file
            img = Image.open(os.path.join(data_dir, class_folder, image_name))
            img = img.resize((224, 224))
            # Convert the image to a numpy array and normalize pixel values
            img_arr = np.array(img) / 255.0
            # Append the image array and label to lists
            images.append(img_arr)
            # print(data_dir,class_folder)
            labels.append(class_folder)
            
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Encode labels
    # label_encoder = LabelEncoder()
    # labels_encoded = label_encoder.fit_transform(labels)
    # labels_encoded = to_categorical(labels_encoded)

    
    return images, labels

data_dir = "data/merge/update1/shrc-ramaih-600-blood" 
data_dir = "data/merge/update1/update/shrc-ramaih-600-blood-1" 
data_dir = "data/6-classes/new-data" 
data_dir = "data/6-classes/new-data/6-c"
data_dir = "data/final/data"
data_dir = "data/final/6-c"
# data_dir = "data/final/4"

# data_dir = "../video-frames/new-data" 
# data_dir = "data/merge/reg"          #2073 after modify but removed confusion data
images, labels = load_images_and_labels(data_dir)
print(labels)
labels1=[]

# for i in labels:
#     if i=="5":
#         labels1.append(5.0)
#     elif i=="4":
#         labels1.append(4.0)
#     elif i=="3":
#         labels1.append(3.0)
#     elif i=="2":
#         labels1.append(2.0)
#     else:
#         labels1.append(1.0)

# labels1=[]
# for i in labels:
#     if i=="5":
#         labels1.append(5.0)
#     elif i=="4":
#         labels1.append(4.0)
#     elif i=="3.5":
#         labels1.append(3.5)
#     elif i=="2":
#         labels1.append(2.0)
#     elif i=='1':
#         labels1.append(1.0)

# for i in labels:
#     if i=="6":
#         labels1.append(6.0)
#     elif i=="5":
#         labels1.append(5.0)
#     elif i=="4":
#         labels1.append(4.0)
#     elif i=="3":
#         labels1.append(3.0)
#     elif i=="2":
#         labels1.append(2.0)
#     else:
#         labels1.append(1.0)



for i in labels:
    if i=="3.5":
        labels1.append(3.5)
    elif i=="5":
        labels1.append(5.0)
    elif i=="4":
        labels1.append(4.0)
    elif i=="3":
        labels1.append(3.0)
    elif i=="2":
        labels1.append(2.0)
    else:
        labels1.append(1.0)





labels1=np.array(labels1)
print(labels1)

print(len(images))
X_train, X_test, y_train, y_test = train_test_split(images, labels1, test_size=0.20, random_state=42)



# data_dir = "data/merge/test1"
# images, labels = load_images_and_labels(data_dir)
# print(labels)
# labels1=[]
# for i in labels:
#     if i=="5":
#         labels1.append(5.0)
#     elif i=="4":
#         labels1.append(4.0)
#     elif i=="3":
#         labels1.append(3.0)
#     elif i=="2":
#         labels1.append(2.0)
#     elif i=="1":
#         labels1.append(1.0)

# labels1=np.array(labels1)
# print(labels1)

# print(len(images))
# X_train1, X_test, y_train1, y_test = train_test_split(images, labels1, test_size=0.99, random_state=42)










from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense



# simple cnn net
def create_cnn(width, height, depth):
    model = Sequential()
    
    # Add convolutional layer with 32 filters of size 3x3
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, depth)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Add another convolutional layer with 64 filters of size 3x3
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten the tensor output by the convolutional layers
    model.add(Flatten())
    
    # Add dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    
    # Output layer for regression: no activation function is used
    model.add(Dense(1))

    # Compile the model using mean squared error as the loss function and the Adam optimization algorithm
    # model.compile(loss='mean_squared_error', optimizer='adam')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    
    return model
# Create the model
model = create_cnn(224, 224, 3)



# vgg16 net

from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.resnet101 import ResNet101
from tensorflow.keras.applications import ResNet101, ResNet152
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

# load ResNet50 model without classification layers
# modelName=ResNet50
base_model = InceptionV3(weights='imagenet', include_top=False)


# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# add a fully-connected layer
# x = Dense(4096, activation='relu')(x)
# x = Dense(2048, activation='relu')(x)
x = Dense(512, activation='relu')(x)
# x = Dense(256, activation='relu')(x)

# and a regression layer -- let's say we're predicting one continuous value
predictions = Dense(1)(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])



# print(model.summary())
# plot_model(model, to_file='model_summary.png', show_shapes=True)

# Compile the model
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mean_absolute_error'])


import os
from tensorflow.keras.callbacks import ModelCheckpoint

# Create a directory to save the model if it doesn't exist
save_dir = 'model'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(save_dir, 'reg-blur-up.h5'),
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)


import torch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ",device)



print(len(images), len(labels1), len(X_train),len(X_test))
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test),callbacks=[model_checkpoint])
# model.save('model/reg-blur-vgg16.h5')





# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.suptitle("InceptionV3", fontsize=14)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.savefig("reg-loss-test.jpg")
# plt.show()



from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model('model/reg-blur-up.h5')

y_pred = model.predict(X_test)
from sklearn.metrics import r2_score
y_test=np.array(y_test)
r2 = r2_score(y_test, y_pred)
print('r2 score for perfect model is', r2)

# print(y_pred,y_test)
c=0
c4=[]
c1=[]
c2=[]
c3=[]
c5=[]

for i,j in zip(y_pred,y_test):
    if j==4:
        c4.append([i[0],j])
    if j==1:
        c1.append([i[0],j])
    if j==2:
        c2.append([i[0],j])
    if j==3:
        c3.append([i[0],j])
    if j==5:
        c5.append([i[0],j])
    x=i
    y=j 
    if i<=1.5:
        i=1.0
    elif i>1.5 and i<=2.5:
        i=2.0
    elif i>2.5 and i<=3.5:
        i=3.0
    elif i>3.5 and i<=4.5:
        i=4.0
    elif i>4.5 and i<=5.5:
        i=5.0
    else:
        i=6.0
    print("predicted, round off, original ",x,i,y)
    if i==j:
        c+=1

print(c, len(y_pred))
# print(c4)
# print(c5)
# print(c3)
# print(c2)
# print(c1)




