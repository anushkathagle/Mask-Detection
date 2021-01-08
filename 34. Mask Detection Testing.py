#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import cv2


# In[2]:


import numpy as np


# In[3]:


model = load_model("Mask_NoMask.h5")


# In[4]:


cam = cv2.VideoCapture(0)

###Detect Face
face_class = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    _, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_class.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (224,224))
        face = img_to_array(face)
        face = np.expand_dims(face, axis = 0)
        face = preprocess_input(face)
        
        text = ""
        colour = (0,0,0)
        (mask, withoutmask) = model.predict(face)[0]
        if mask > withoutmask:
            text = 'Mask Detected'
            colour = (0,225,0)
        else:
            text = 'No Mask'
            colour = (0,0,225)
        
        
        cv2.rectangle(img, (x, y), (x+w, y+h), colour, 2)
        cv2.putText(img , text, (x, y -15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    
    cv2.imshow('My Mask Detector', img)
    if cv2.waitKey(1) == 13:
        break
        
cam.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




