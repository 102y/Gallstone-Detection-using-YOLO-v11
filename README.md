# Gallstone Detection using YOLO v11
Project Description
This project aims to develop a YOLO v11 model for accurately detecting the locations of gallstones in gallbladder images. The project is based on the dataset "Gallstone detection by using YOLOV 11 - AMLESH THAKUR" available through Roboflow. The ultralytics library is used to load, train, and analyze images with the model.
Project Requirements
Python
The ultralytics library for YOLO
A dataset in the specified path
Core Code
Loading and Training the Model:
from ultralytics import YOLO

# Load the model
model = YOLO("C://Users//NITRO//Desktop//AI Proj//LungCancer//yolo11n-seg.pt")

# Train the model
train_results = model.train(
    data="C://Users//NITRO//Desktop//AI Proj//LungCancer//Gallstone detection by using YOLOV 11 -AMLESH THAKUR-.v1i.yolov11//data.yaml",
    epochs=10,
    imgsz=640,
    device="cpu"
)
Making Predictions on New Images:
python
نسخ الكود
# Load the trained model
model = YOLO('runs//segment//train//weights//best.pt')

# Test the model on a single image
results = model("C://Users//NITRO//Desktop//AI Proj//LungCancer//Gallstone detection by using YOLOV 11 -AMLESH THAKUR-.v1i.yolov11//test//images//3vesselsbladder_112_jpg.rf.df7b63aaea4988133c947eb798bbdd45.jpg", save=True)
results[0].show()

# Test the model on a batch of images
results = model("test_images", save=True)
Notes:
Ensure the data.yaml file in the dataset contains the correct paths for images and class names.
The number of epochs and training configuration can be adjusted as needed for the project.
