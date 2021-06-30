# Facial_emotion_recognition

This model detect and classify facial emotion of five categories (neutral, anger, surprise, smile and sad)



# Approach
This model are composed two part: Facial detection and emotion classification
- Facial detector: I use dlib library with pretrained network("mmod_human_face_detector") to dectect face. Network can be downloaded from [here](http://dlib.net/files/).

- Emotion classifier: To make more light-weight model, I selected MobileNetV2 as a classifier.

# Technique
- Transfer Learning : Before training on these dataset, I have trained MobileNetV2 on [CK+ dataset](https://www.kaggle.com/shawon10/ckplus), which is 48x48 face image and have 7 emotion category. By this way, I tried to find better initial weight and give more stable training to model.
