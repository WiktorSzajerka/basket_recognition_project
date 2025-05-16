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

model_path = "enter_your_path_here"
cnn_weights_path = "enter_your_path_here"

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

        threading.Thread(target=self.procesing_video, daemon=True).start()

    def procesing_video(self):
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

        

        model = YOLO(model_path)
        model2 = Net()
        model2.load_state_dict(torch.load(cnn_weights_path, weights_only=True))
        model2.eval()
        cap = cv2.VideoCapture(self.video_path)
        i = 0
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
                        if list[0][2] - list[0][0] < 80:
                            cr_im, _ = crop_basket(list, frame)
                            with torch.no_grad():
                                output = model2(cr_im)
                                _, predicted = torch.max(output, 1)
                                klasa = predicted.item()
                            
                            if klasa == 1:
                                i = 160
                                frame = cv2.putText(frame, 'Shot made', org=(320, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                    fontScale=1, color=(0,255,0), thickness=2)
                    cv2.imshow("w", frame)
                    cv2.waitKey(120)
            else:
                i -= 1
                cv2.imshow("w", frame)
                cv2.waitKey(5)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = BasketballRecognitionApp(root)
    root.mainloop()




