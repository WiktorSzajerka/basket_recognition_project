from ultralytics import YOLO
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import threading


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

class BasketballRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Basketball Recognition Model")
        self.root.geometry("600x400")
        self.root.configure(bg="#f0f0f0")


        self.label = tk.Label(root, text="Video Path:", bg="#f0f0f0", font=("Helvetica", 12))
        self.label.pack(pady=10)

        self.video_path_entry = tk.Entry(root, width=50, font=("Helvetica", 12))
        self.video_path_entry.pack(pady=10)

        self.browse_button = tk.Button(root, text="Browse", command=self.browse_file, bg="#f0f0f0", fg="black", font=("Helvetica", 12))
        self.browse_button.pack(pady=10)

        self.play_button = tk.Button(root, text="Play Video", command=self.play_video, bg="#2196f3", fg="white", font=("Helvetica", 12))
        self.play_button.pack(pady=20)

        

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            self.video_path_entry.delete(0, tk.END)
            self.video_path_entry.insert(0, file_path)

    def play_video(self):
        self.video_path = self.video_path_entry.get()
        if not self.video_path:
            messagebox.showerror("Error", "Please enter or select a video path.")
            return

        threading.Thread(target=self.precesing_video, daemon=True).start()

    def precesing_video(self):
        def crop_basket(bbox, frame):
            list = bbox[0]
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
            if l < 250:
                side ='l'
            else:
                side = 'r'
            cr_im = frame[t:b, l:r]
            cr_im = transformation(cr_im)
            cr_im = cr_im.unsqueeze(0)
            return cr_im, side

        def line_extr(image, lr):
            result = np.zeros_like(image)
            coordinates = []
            if lr == 'l':
                for y in range(image.shape[0]):
                    row = image[y, :]
                    leftmost_idx = np.argmax(row)
                    if row[leftmost_idx] > 0: 
                        result[y, leftmost_idx] = 255
                        coordinates.append(leftmost_idx)
                    else:
                        coordinates.append(image.shape[1])
            else:
                for y in range(image.shape[0]):
                    row = image[y, :]
                    rightmost_idx = len(row) - 1 - np.argmax(row[::-1])
                    if row[rightmost_idx] > 0:  
                        result[y, rightmost_idx] = 255
                        coordinates.append(rightmost_idx)
                    else:
                        coordinates.append(0)

            return result, coordinates

        def check_3pt(line, ankles, side):
            if side == 'r':
                for ank in ankles:
                    x = 0
                    for i in range(int(ank[1])-5, int(ank[1])+5):
                        dist = line[i] - ank[0]
                        if dist < 7:
                            x += 1
                    if x > 2:
                        return False
            else:
                for ank in ankles:
                    x = 0
                    for i in range(int(ank[1])-5, int(ank[1])+5):
                        dist = ank[0] - line[i]
                        if dist < 7:
                            x += 1
                    if x > 2:
                        return False
                    
            return True

        def shoot_det(player_head, hand, ballxy, side):
            x = player_head[0] - ballxy[0]
            print('x ', x)
            y = player_head[1] - ballxy[1]
            print('y ', y)
            hand_y = hand[1] - ballxy[1]
            print('hand_y ', hand_y)
            hand_head = hand[1] - player_head[1]
            print('hand_head', hand_head)
            if y > -2 and hand_y > 0 and hand_head > -3:
                if side == 'l':
                    if x > -3 and x < 30:
                        return True
                elif side == 'r':
                    if x < 3 and x > -30:
                        return True
            
            return False


        model = YOLO(r'app\best.pt')
        model_pose = YOLO(r'app\yolov8s-pose.pt')
        model2 = Net()
        model2.load_state_dict(torch.load(r'app\basket_cl.pt', weights_only=True))
        model2.eval()
        cap = cv2.VideoCapture(self.video_path)
        ball_center = [0, 0]
        three = False
        i = 0
        j = 0
        shoot = False
        cnt = 0
        white = False
        while True:
            ret, frame1 = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame1, (640, 360))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if i == 0:
                results = model.predict(source=frame)
                ball_center = [0, 0]
                for result in results:
                    if 1 in result.boxes.cls and 0 in result.boxes.cls:
                        for box in result.boxes:
                            cl = box.cls[0]
                            if cl == 1:
                                list = box.xyxy.tolist()
                                if len(list) != 0:
                                    if list[0][2] - list[0][0] < 80:
                                        cr_im, side = crop_basket(list, frame)
                                        with torch.no_grad():
                                            output = model2(cr_im)
                                            _, predicted = torch.max(output, 1)
                                            klasa = predicted.item()
                                        if klasa == 1:
                                            i = 160
                                
                            elif cl == 0:
                                list = box.xyxy.tolist()
                                if len(list) != 0:
                                    list = list[0]
                                    l, t, r, b = list
                                    ball_center = [l+(r - l)/2, t+(b - t)/2]

                    elif 1 in result.boxes.cls:
                        for box in result.boxes:
                            cl = box.cls[0]
                            if cl == 1:
                                list = box.xyxy.tolist()
                                if len(list) != 0:
                                    if list[0][2] - list[0][0] < 80:
                                        cr_im, side = crop_basket(list, frame)
                                        with torch.no_grad():
                                            output = model2(cr_im)
                                            _, predicted = torch.max(output, 1)
                                            klasa = predicted.item()
                                        if klasa == 1:
                                            i = 160

                        ball_img = frame1
                        if j == 0:
                            if shoot and cnt < 10:
                                if prev_ball_center[1] > ball_center[1] :
                                    cnt += 1
                                else:
                                    shoot = False
                                    cnt = 0
                            
                            elif shoot == False and ball_center[0] != 0:
                                results_p = model_pose.predict(source=frame, conf=0.5)
                                for result in results_p:
                                    keypoints = result.keypoints.xy.tolist()
                                    dist = 999
                                    for ind, player in enumerate(keypoints):
                                        dist1 = abs(ball_center[0] - player[9][0])
                                        dist1 += abs(ball_center[1] - player[9][1])
                                        dist1 += abs(ball_center[0] - player[10][0])
                                        dist1 += abs(ball_center[1] - player[10][1])
                                        if dist1 < dist:
                                            dist = dist1
                                            nr = ind
                                    
                                    player_kp = keypoints[nr]
                                    ind = 0
                                    while player_kp[ind][0] == 0 and ind < 15:
                                        ind += 1

                                    shoot = shoot_det(player_kp[ind], player_kp[9], ball_center, side)                              
                                    ankles = [player_kp[15], player_kp[16]]
                                    prev_ball_center = ball_center
                                    if white:
                                        ret,img = cv2.threshold(img,245,255,0)
                                    else:
                                        img = cv2.bitwise_not(img)
                                        ret,img = cv2.threshold(img,235,255,0)
                                    processed_image, line = line_extr(img, side)
                                    cnt = 1
                                    if shoot:
                                        three = check_3pt(line, ankles, side)
                                        print(f'Za trzy: {three}')
                            
                                                                

                        if cnt == 10:
                            if three:
                                frame = cv2.putText(frame, 'rzut za 3pt', org=(330, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                    fontScale=1, color=(0,255,0), thickness=2)
                            else:
                                frame = cv2.putText(frame, 'rzut za 2pt', org=(330, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                    fontScale=1, color=(0,255,0), thickness=2)
                            j = 120
                            shoot = False
                            cnt = 0
                

                    print('shoot', shoot)
                    print(j)
                    cv2.imshow("w", frame)
                    cv2.waitKey(180)
                if j != 0:
                    j -= 1
                    
            else:
                i -= 1
                j = 0
                shoot = False
                if three:
                    frame = cv2.putText(frame, 'trafiony za 3pt', org=(330, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1, color=(0,255,0), thickness=2)
                else:
                    frame = cv2.putText(frame, 'trafiony za 2pt', org=(330, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1, color=(0,255,0), thickness=2)
                cv2.imshow("w", frame)
                cv2.waitKey(5)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = BasketballRecognitionApp(root)
    root.mainloop()




