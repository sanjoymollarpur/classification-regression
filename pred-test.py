import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sns
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

data_dir = "data/merge/reg-update-1" #2267 after modify
data_dir = "data/merge/reg-update-2"
data_dir = "data/merge/reg-update-2/added1"
data_dir = "data/merge/reg-update-2/added2"
data_dir = "data/merge/update1/shrc-ramaih-600-blood" 
data_dir = "data/merge/update1/update/shrc-ramaih-600-blood-1" 
data_dir = "data/6-classes/new-data" 
data_dir = "data/6-classes/new-data/6-c"
data_dir = "data/final/data"
data_dir = "data/verified-data/5c-train-test/test"
# data_dir = "data/final/4"

# data_dir = "../video-frames/new-data" 
# data_dir = "data/merge/reg"          #2073 after modify but removed confusion data
images, labels = load_images_and_labels(data_dir)
print(labels)
labels1=[]



labels1=[]
for i in labels:
    if i=="5":
        labels1.append(5.0)
    elif i=="4":
        labels1.append(4.0)
    elif i=="3":
        labels1.append(3.0)
    elif i=="3.5":
        labels1.append(3.5)
    elif i=="2":
        labels1.append(2.0)
    elif i=='1':
        labels1.append(1.0)



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





labels1=np.array(labels1)
print(labels1)

print(len(images))
X_train, X_test, y_train, y_test = train_test_split(images, labels1, test_size=0.99, random_state=42)


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model('model/reg-blur-5c.h5')

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

# print(X_test)

# for i,j in zip(y_pred,y_test):
#     if i>=3.3 and i<=4.0:
#         c1.append([i[0],j])
#     if i>=4.01 and i<=4.5:
#         c2.append([i[0],j])

#     x=i
#     y=j 
#     if i<=1.5:
#         i=1.0
#     elif i>1.5 and i<=2.5:
#         i=2.0
#     elif i>2.5 and i<=3.5:
#         i=3.0
#     elif i>3.5 and i<=4.5:
#         i=4.0
#     elif i>4.5 and i<=5.5:
#         i=5.0
#     else:
#         i=6.0
#     print("predicted, round off, original ",x,i,y)
#     if i==j:
#         c+=1

true_classes=[]
predicted_classes=[]


for i,j in zip(y_pred,y_test):
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
    true_classes.append(j)
    predicted_classes.append(i)

    print("predicted, round off, original ",x,i,y)
    if i==j:
        c+=1


# for i,j in zip(y_pred,y_test):
#     x=i
#     y=j 
#     if i<=1.5:
#         i=1.0
#     elif i>1.5 and i<=2.5:
#         i=2.0
#     elif i>2.5 and i<=3.2:
#         i=3.0
#     elif i>3.2 and i<=3.8:
#         i=3.5
#     elif i>3.8 and i<=4.8:
#         i=4.0
#     elif i>4.8 and i<=5.5:
#         i=5.0
#     if i==3.5:
#         i=6.0
#     if j==3.5:
#         j=6.0
#     true_classes.append(j)
#     predicted_classes.append(i)

#     print("predicted, round off, original ",x,i,y)
#     if i==j:
#         c+=1


print(c, len(y_pred))
print(true_classes)
print(predicted_classes)


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)


accuracy = accuracy_score(true_classes, predicted_classes)
# precision = precision_score(true_classes, predicted_classes)
# recall = recall_score(true_classes, predicted_classes)
# f1 = f1_score(true_classes, predicted_classes)

print("acc: ",accuracy)


plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.savefig("cm-matrix-3.png")
# plt.show()
plt.close()

