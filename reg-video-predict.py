
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model('model/reg-blur-89.h5')
import numpy as np

import glob 
import cv2



# import cv2
# import numpy as np
# from keras.models import load_model


url="668715_1542_1-test-89"

# Open the video file
# video_path = 'data/video/clip/649960_201_2-825-835.mp4'  # Replace with your video file
# video_path = f'data/predict-video/clip/{url}.mp4'
video_path = f'test-video/668715_1542_1.avi'
cap = cv2.VideoCapture(video_path)

# Preprocess parameters (e.g., resizing, normalization)
target_size = (224, 224)
normalize = True

# Initialize predictions list
predictions = []

c=0
# Iterate over frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    xx=frame
    img=xx
    # img = img.resize((64, 64))
    frame = cv2.resize(frame, target_size)
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    
    # Make prediction on the frame
    prediction = model.predict(frame)
    # predictions.append(prediction)

    
    prediction=str(prediction[0][0])[:5]
    print(prediction)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 110)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    xx = cv2.putText(xx, prediction, org, font, fontScale, color, thickness, cv2.LINE_AA)
    
    # print(r)
    filename=f"data/predict-video/c-3/{c}.jpg"
    cv2.imwrite(filename, xx)
    c+=1



    
# # Convert predictions list to numpy array
# predictions = np.concatenate(predictions)

# # Aggregate predictions (e.g., averaging)
# final_prediction = np.mean(predictions, axis=0)

# # Print the final prediction
# print(final_prediction)




import cv2
import os

path = 'data/predict-video/c-3'  # Directory containing your image frames
output = f'{url}.avi'  # Output video filename
fps = 11  # Frames per second
width = 224  # Width of the video frames
height = 224  # Height of the video frames


frame_files = sorted(os.listdir(path))
frame_files.sort()
print(frame_files)
a=[]
for i in range(0,len(frame_files)):
    p=f"{i}.jpg"
    a.append(p)
frame_files=a
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

# Iterate through the frames and add them to the video
for frame_file in frame_files:
    frame_path = os.path.join(path, frame_file)
    frame = cv2.imread(frame_path)
    print(frame_path)
    # Resize the frame if necessary
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height))
    
    # Write the frame to the video
    video_writer.write(frame)

# Release the VideoWriter object

video_writer.release()

print(f"Video saved to {output}")















# p=[]
# for i in glob.glob("data/video/18553_4564_1.avi"):
#     # p.append(i)


#     # path='data/video/22_3507_1.avi'
#     path=i
#     path_name=i.split('/')[-1].split('.')[0]
#     SIZE = 224

#     ct=[]
#     cl=[]
#     cou=0

#     vidcap = cv2.VideoCapture(path)
#     success, img = vidcap.read()
#     #print(img)
#     c=0
#     while success and vidcap.isOpened():
#         success, image1 = vidcap.read()
#         xx=image1
#         if success and c%1==0:
#             img_path=image1    
#             # print(img_path, c)

#             # img = cv2.imread(img_path, 0)
#             # img=cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY)
#             # img = cv2.resize(img, (SIZE, SIZE))

#             test_image = image.load_img(img_path, target_size=(224, 224))
#             test_image = image.img_to_array(test_image)
#             test_image = np.expand_dims(test_image, axis=0)
#             test_image = test_image / 255.0

#             # Make the prediction
#             classes = model.predict(test_image)
#             y_pred = np.argmax(classes, axis=1)


#             if y_pred[0] == 1:
#                 prediction = 'clear'

#                 r=[{ "image" : prediction}]
#             elif result[0]==0:
#                 prediction = 'blur'
#                 r= [{ "image" : prediction}]
    
#             font = cv2.FONT_HERSHEY_SIMPLEX
  
#             # org
#             org = (50, 50)
#             # fontScale
#             fontScale = 1
#             # Blue color in BGR
#             color = (255, 0, 0)
#             # Line thickness of 2 px
#             thickness = 2
#             # Using cv2.putText() method
#             xx = cv2.putText(xx, prediction, org, font, fontScale, color, thickness, cv2.LINE_AA)
#             print(r)
#             filename=f"data/predict-video/{prediction}/{path_name}-{c}.jpg"
#             cv2.imwrite(filename, xx)
