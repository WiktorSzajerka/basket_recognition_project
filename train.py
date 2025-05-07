from ultralytics import YOLO

model = YOLO(r'runs\detect\basket_det4\weights\last.pt')
 

results = model.train(
   data='data.yaml',
   imgsz=640,
   epochs=8,  
   batch=6,
   name='basket_det'
)