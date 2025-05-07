from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator 
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
class Net(nn.Module):

    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5)
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
        
        self.pool = nn.MaxPool2d(2, 2)
       
        self.fc1 = nn.Linear(5 * 5 * 24, out_features=120)

        self.fc2 = nn.Linear(120, out_features=num_classes)

    def forward(self, x):
      
        x = F.relu(self.pool(self.conv1(x))) 
        x = F.relu(self.pool(self.conv2(x)))  
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)
    
transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


model = YOLO(r'runs\detect\basket_det2\weights\best.pt')
model2 = Net()
model2.load_state_dict(torch.load('basket_cl.pt', weights_only=True))
model2.eval()
cap = cv2.VideoCapture(r"videos\test\Stewart Tip Layup Shot (2 PTS).mp4")
l, t, r, b = 0, 0, 640, 384
i = 0
j = 4
while True:
    ret, frame1 = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame1, (640, 384))

    if i == 0:
        results = model.predict(source=frame)
        for result in results:
            list = result.boxes.xyxy.tolist()
            if len(list) != 0:
                list = list[0]
                if list[2] - list[0] < 45:
                    list = [int(round(x)) for x in list]
                    l, t, r, b = list
                    width = r - l
                    height = b - t
                    if width < 32:
                        l = l - int(np.ceil((32 - width)/2))
                        r = r + int(np.floor((32 - width)/2))
                    elif width > 32:
                        l = l + int(np.ceil((width - 32)/2))
                        r = r - int(np.floor((width - 32)/2))
                    if height > 32:
                        t = t + int(np.ceil((height - 32)/2))
                        b = b - int(np.floor((height - 32)/2))
                    elif height < 32:
                        t = t - int(np.ceil((32 - height)/2))
                        b = b + int(np.floor((32 - height)/2))
            
                    cr_im = frame[t:b, l:r]
                    cr_im = transformation(cr_im)
                    cr_im = cr_im.unsqueeze(0)
                    with torch.no_grad():
                        output = model2(cr_im)
                        _, predicted = torch.max(output, 1)
                        klasa = predicted.item()
                        print("Przewidywana klasa", klasa)
                    if klasa == 1:
                        i = 160
                        frame = cv2.putText(frame, 'Trafiony', org=(320, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=1, color=(0,255,0), thickness=2)
            cv2.imshow("w", frame)
            cv2.waitKey(120)
    else:
        i -= 1
        cv2.imshow("w", frame)
        cv2.waitKey(5)