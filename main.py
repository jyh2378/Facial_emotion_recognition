import time
import copy
import numpy as np
import dlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from misc import *
from data_loader import FacialDataLoader, CKDataLoader
from graphs import ResNet18, mobilenet_v2


class Agent:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = mobilenet_v2(num_classes=7)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=5e-4)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.9, last_epoch=-1)
        self.data_loader = FacialDataLoader(batch_size=128)
        self.train_acc_history, self.val_acc_history = [], []

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def train(self, num_epochs):
        self.model.to(self.device)
        for epoch in range(num_epochs):
            if epoch > 20:
                self.lr_scheduler.step()
            for group in self.optimizer.param_groups:
                now_lr = group['lr']
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print(f'lr is {now_lr}')
            print('-' * 10)
            self.train_per_epoch()
            self.validate()

            if epoch < 10:
                continue
            if self.val_acc_history[-1] == max(self.val_acc_history):
                print(f"model saved at epoch")
                state = {'model_state_dict': self.model.state_dict(),
                         }
                if not os.path.isdir("checkpoints"):
                    os.mkdir("checkpoints")
                torch.save(state, os.path.join("checkpoints", 'mobilenet_v2(' + str(self.val_acc_history[-1].item()) +').pth'))

        print(f"best val acc {max(self.val_acc_history)}")

    def train_per_epoch(self):
        train_loss, total, correct = 0., 0, 0
        self.model.train()
        for batch_idx, (inputs, labels) in enumerate(self.data_loader.train_loader):
            self.optimizer.zero_grad()

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print(f"\rtrain loss: {train_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f} ({correct}/{total})", end='')
        print(f"train loss: {train_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f} ({correct}/{total})")
        self.train_acc_history.append(100. * correct / total)

    def validate(self):
        val_loss, total, correct = 0., 0, 0
        self.model.to(self.device)
        self.model.eval()
        for batch_idx, (inputs, labels) in enumerate(self.data_loader.valid_loader):
            self.optimizer.zero_grad()

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()

            print(f"\rval loss: {val_loss/(batch_idx+1):.3f} | Acc: {100.*correct / total:.3f} ({correct}/{total})", end='')
        print(f"val loss: {val_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f} ({correct}/{total})")

        self.val_acc_history.append(100.*correct/total)

    def face_dectector(self, img_path):
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cnnFaceDetector = dlib.cnn_face_detection_model_v1("model/mmod_human_face_detector.dat")
        rects = cnnFaceDetector(img_gray, 1)
        return img, img_gray, rects

    def test(self):
        test_dir = "datasets/facial_emotion_data/test/img"
        predict_dir = "datasets/facial_emotion_data/test/predict"       # 예측 bbox와 label을 이미지에 그린 다음 저장

        test_img_name_list = os.listdir(test_dir)
        label_dic = {0: "anger", 1: "neutral", 2: "sad", 3: "smile", 4: "surprise"}
        for idx, img_name in enumerate(test_img_name_list):
            img_path = os.path.join(test_dir, img_name)
            start = time.time()
            img, img_gray, rects = self.face_dectector(img_path)
            print(f"{idx}th img face detect time:{time.time() - start}")
            for (i, rect) in enumerate(rects):
                x1 = rect.rect.left()
                y1 = rect.rect.top()
                x2 = rect.rect.right()
                y2 = rect.rect.bottom()
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                face = img[y1:y2, x1:x2]
                start = time.time()
                face = cv2.resize(face, dsize=(112, 112)).transpose(2, 0, 1)
                face = torch.FloatTensor(face).unsqueeze(0)
                pred = torch.argmax(self.model(face)).item()
                pred_label = label_dic[pred]
                cv2.putText(img, str(pred_label), (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
                print(f"{idx}th img {i}th face classifying time:{time.time() - start}")

            cv2.imwrite(os.path.join(predict_dir, img_name), img)


if __name__ == '__main__':
    chk_path = "checkpoints/mobilenet_v2(63).pth"
    agent = Agent(chk_path)
    agent.test()
