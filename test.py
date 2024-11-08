from ultralytics import YOLO

# Load a model
model = YOLO("C://Users//NITRO//Desktop//AI Proj//LungCancer//yolo11n-seg.pt")

# Train the model
train_results = model.train(
    data="C://Users//NITRO//Desktop//AI Proj//LungCancer//Gallstone detection by using YOLOV 11 -AMLESH THAKUR-.v1i.yolov11//data.yaml",  # path to dataset YAML
    epochs=10,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

model = YOLO('runs//segment//train//weights//best.pt')
results = model("C://Users//NITRO//Desktop//AI Proj//LungCancer//Gallstone detection by using YOLOV 11 -AMLESH THAKUR-.v1i.yolov11//test//images//3vesselsbladder_112_jpg.rf.df7b63aaea4988133c947eb798bbdd45.jpg", save=True)
results[0].show()

model = YOLO('runs//segment//train//weights//best.pt')
results = model("test_images", save=True)