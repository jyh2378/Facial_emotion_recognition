import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class FaceDataset():
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "train\\img"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "train\\annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "train\\img", self.imgs[idx])
        ant_path = os.path.join(self.root, "train\\annotations", self.annotations[idx])
        img = Image.open(img_path).convert("RGB")

        tree = ET.parse(ant_path)
        root = tree.getroot()
        objects = root.findall('object')
        dic_labels = {"neutral": 0, "anger": 1, "surprise": 2, "smile": 3, "sad": 4}
        boxes, labels = [], []
        for i, obj in enumerate(objects):
            label = str(obj.find('name').text)

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            bbox = (xmin, ymin, xmax, ymax)
            boxes.append(bbox)
            labels.append(dic_labels[label])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 6
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280
anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

def get_transform(train):
    transform_list = []
    transform_list.append(transforms.ToTensor())
    if train:
        transform_list.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transform_list)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
dataset = FaceDataset('datasets\\facial_emotion_data', train_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

images, targets = next(iter(data_loader))

###############################

train_transform = transforms.Compose([
    # transforms.Grayscale(),
    transforms.TenCrop((44, 44)),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

dataset = datasets.ImageFolder('datasets/CK+48', transform=train_transform)
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=0,
    shuffle=True
)

it = iter(loader)
data, label = next(it)
plt.imshow(data[0].numpy().transpose((1,2,0)))
print(label)


mean = 0.
std = 0.
nb_samples = 0.
for data, _ in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

###############################
import matplotlib.pyplot as plt
import dlib
import cv2
from imutils import face_utils

path = 'datasets\\facial_emotion_data\\train\\img'
img_path = 'datasets\\facial_emotion_data\\train\\img\\7.14leis.jpg'

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_detect = dlib.get_frontal_face_detector()
dnnFaceDetector = dlib.cnn_face_detection_model_v1("trained_model/mmod_human_face_detector.dat")
rects = face_detect(img_gray, 1)
for (i, rect) in enumerate(rects):
    x1 = rect.left()
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
plt.imshow(img)
plt.show()
