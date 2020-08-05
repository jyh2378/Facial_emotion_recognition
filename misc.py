import os
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import skimage.io
import h5py


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def preprocess_face_data(img_size=(48, 48), to_gray=False):
    """facial emotion 데이터의 annotation을 읽어, 얼굴 crop & resize를 거쳐 label/img 형태의 dir로 정리합니다."""
    origin_train_dir = 'datasets\\facial_emotion_data\\train\\img'
    train_annotation_dir = 'datasets\\facial_emotion_data\\train\\annotations'

    extracted_dir = 'datasets\\facial48'

    img_name_list = os.listdir(origin_train_dir)
    annot_name_list = os.listdir(train_annotation_dir)

    for img_name, annot_name in zip(img_name_list, annot_name_list):
        img_path, annot_path = os.path.join(origin_train_dir, img_name), os.path.join(train_annotation_dir, annot_name)

        tree = ET.parse(annot_path)
        root = tree.getroot()
        objects = root.findall('object')
        for i, obj in enumerate(objects):
            img = cv2.imread(img_path)

            label = str(obj.find('name').text)

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            img = img[ymin:ymax, xmin:xmax]

            if to_gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img.shape[0] < img_size[0] or img.shape[1] < img_size[1]:    # img가 확대될 경우
                img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_LINEAR)
            else:                                                           # img가 축소될 경우
                img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_AREA)

            new_img_name = img_name.strip('.jpg') + '_' + str(i) + '.jpg'
            cv2.imwrite(os.path.join(extracted_dir, 'train', label, new_img_name), img)


ck_path = "datasets/CK+48"
def ck_to_h5py():
    """CK+48 데이터를 h5py 파일로 묶어 저장합니다."""
    anger_path = os.path.join(ck_path, 'anger')
    disgust_path = os.path.join(ck_path, 'disgust')
    fear_path = os.path.join(ck_path, 'fear')
    happy_path = os.path.join(ck_path, 'happy')
    sadness_path = os.path.join(ck_path, 'sadness')
    surprise_path = os.path.join(ck_path, 'surprise')
    contempt_path = os.path.join(ck_path, 'contempt')

    # # Creat the list to store the data and label information
    data_x = []
    data_y = []

    datapath = os.path.join('datasets', 'CK_data.h5')
    if not os.path.exists(os.path.dirname(datapath)):
        os.makedirs(os.path.dirname(datapath))

    # order the file, so the training set will not contain the test set (don't random)
    files = os.listdir(anger_path)
    files.sort()
    for filename in files:
        I = skimage.io.imread(os.path.join(anger_path, filename))
        data_x.append(I.tolist())
        data_y.append(0)

    files = os.listdir(disgust_path)
    files.sort()
    for filename in files:
        I = skimage.io.imread(os.path.join(disgust_path, filename))
        data_x.append(I.tolist())
        data_y.append(1)

    files = os.listdir(fear_path)
    files.sort()
    for filename in files:
        I = skimage.io.imread(os.path.join(fear_path, filename))
        data_x.append(I.tolist())
        data_y.append(2)

    files = os.listdir(happy_path)
    files.sort()
    for filename in files:
        I = skimage.io.imread(os.path.join(happy_path, filename))
        data_x.append(I.tolist())
        data_y.append(3)

    files = os.listdir(sadness_path)
    files.sort()
    for filename in files:
        I = skimage.io.imread(os.path.join(sadness_path, filename))
        data_x.append(I.tolist())
        data_y.append(4)

    files = os.listdir(surprise_path)
    files.sort()
    for filename in files:
        I = skimage.io.imread(os.path.join(surprise_path, filename))
        data_x.append(I.tolist())
        data_y.append(5)

    files = os.listdir(contempt_path)
    files.sort()
    for filename in files:
        I = skimage.io.imread(os.path.join(contempt_path, filename))
        data_x.append(I.tolist())
        data_y.append(6)

    print(np.shape(data_x))
    print(np.shape(data_y))

    datafile = h5py.File(datapath, 'w')
    datafile.create_dataset("data_pixel", dtype='uint8', data=data_x)
    datafile.create_dataset("data_label", dtype='int64', data=data_y)
    datafile.close()

    print(f"Save data finish!!!")


face_path = "datasets/facial48/train"
def facial_to_h5py():
    """facial emotion 데이터를 h5py 파일로 묶어 저장합니다."""
    anger_path = os.path.join(face_path, 'anger')
    neutral_path = os.path.join(face_path, 'neutral')
    sad_path = os.path.join(face_path, 'sad')
    smile_path = os.path.join(face_path, 'smile')
    surprise_path = os.path.join(face_path, 'surprise')

    # # Creat the list to store the data and label information
    data_x = []
    data_y = []

    datapath = os.path.join('datasets', 'facial48.h5')
    if not os.path.exists(os.path.dirname(datapath)):
        os.makedirs(os.path.dirname(datapath))

    # order the file, so the training set will not contain the test set (don't random)
    files = os.listdir(anger_path)
    files.sort()
    for filename in files:
        I = skimage.io.imread(os.path.join(anger_path, filename))
        data_x.append(I.tolist())
        data_y.append(0)

    files = os.listdir(neutral_path)
    files.sort()
    for filename in files:
        I = skimage.io.imread(os.path.join(neutral_path, filename))
        data_x.append(I.tolist())
        data_y.append(1)

    files = os.listdir(sad_path)
    files.sort()
    for filename in files:
        I = skimage.io.imread(os.path.join(sad_path, filename))
        data_x.append(I.tolist())
        data_y.append(2)

    files = os.listdir(smile_path)
    files.sort()
    for filename in files:
        I = skimage.io.imread(os.path.join(smile_path, filename))
        data_x.append(I.tolist())
        data_y.append(3)

    files = os.listdir(surprise_path)
    files.sort()
    for filename in files:
        I = skimage.io.imread(os.path.join(surprise_path, filename))
        data_x.append(I.tolist())
        data_y.append(4)

    print(np.shape(data_x))
    print(np.shape(data_y))

    datafile = h5py.File(datapath, 'w')
    datafile.create_dataset("data_pixel", dtype='uint8', data=data_x)
    datafile.create_dataset("data_label", dtype='int64', data=data_y)
    datafile.close()
    print(f"Save data finish!!!")

if __name__ == "__main__":
    # ck_to_h5py()
    preprocess_face_data((128, 128), to_gray=True)
    facial_to_h5py()
