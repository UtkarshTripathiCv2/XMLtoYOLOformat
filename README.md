# XMLtoYOLOformat


Instructable: Training a Custom YOLOv8 Object Detector
This guide provides a full, step-by-step walkthrough for training a custom YOLOv8 object detection model. We will cover the entire pipeline, from annotating images in PASCAL VOC (.xml) format to converting them to YOLO (.txt), splitting the data, training the model in Google Colab, and finally, using it for real-time detection in a local environment.

Table of Contents
Prerequisites

Step 1: Data Annotation (XML Format)

Step 2: Converting XML to YOLO .txt Format

Step 3: Splitting the Dataset for Training

Step 4: Creating the data.yaml File

Step 5: Training the Model in Google Colab

Step 6: Real-Time Detection with VS Code

1. Prerequisites
Before you begin, ensure you have the following installed on your local machine:

Python 3.8+: Download from python.org.

An IDE: Visual Studio Code is recommended.

Annotation Tool: LabelImg is a great free tool for creating .xml annotations.

2. Step 1: Data Annotation (XML Format)
The first step is to collect images and annotate them. When you draw bounding boxes around objects, your annotation tool will save the labels.

Action: Use a tool like LabelImg to draw boxes around the objects in your images.

Output Format: Set the output format to PASCAL VOC. This will create one .xml file for each .jpg image.

Result: You will have a folder containing pairs of image_name.jpg and image_name.xml files.

3. Step 2: Converting XML to YOLO .txt Format
YOLO models don't use .xml files. They require a specific .txt format. We'll use a Python script to convert our annotations.

The Conversion Script
Create a file named convert.py in your dataset folder and paste the following code into it.

Python

import glob
import os
import xml.etree.ElementTree as ET

# --- Configuration ---
# Your class names, in the order you want them to be indexed (0, 1, 2, ...)
# This MUST match the order in your final data.yaml and classes.txt
class_names = ["MyClassName1", "MyClassName2"]
# ---------------------

def convert_voc_to_yolo(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def convert_annotations(path):
    for xml_file in glob.glob(path + "/*.xml"):
        txt_file_path = os.path.splitext(xml_file)[0] + ".txt"
        with open(txt_file_path, "w") as out_file:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find("size")
            w = int(size.find("width").text)
            h = int(size.find("height").text)

            for obj in root.iter("object"):
                cls = obj.find("name").text
                if cls not in class_names:
                    continue
                cls_id = class_names.index(cls)
                xmlbox = obj.find("bndbox")
                b = (
                    float(xmlbox.find("xmin").text),
                    float(xmlbox.find("xmax").text),
                    float(xmlbox.find("ymin").text),
                    float(xmlbox.find("ymax").text),
                )
                bb = convert_voc_to_yolo((w, h), b)
                out_file.write(f"{cls_id} {' '.join(map(str, bb))}\n")
    print("Conversion complete.")

# Run the conversion on the current folder
convert_annotations('.')
How to Use
Place the convert.py script inside your folder with all the .jpg and .xml files.

Crucially, edit the class_names list at the top of the script to match your project's classes.

Open a terminal in this folder and run: python convert.py.

⚠️ Common Problems & Solutions
Problem: The command ren *.xml *.txt (or similar) doesn't work; it says "file not found".

Cause: You cannot simply rename .xml files. The internal format is completely different.

Solution: You must use a conversion script like the one provided above to read the coordinates from the XML and write them in the correct YOLO format.

Problem: Running python convert.py gives the error [Errno 2] No such file or directory.

Cause: This is almost always a hidden file extension issue on Windows. You likely saved the script as convert.py.txt.

Solution: In Windows File Explorer, go to View and check the box for "File name extensions". Then, rename your file from convert.py.txt to convert.py.

4. Step 3: Splitting the Dataset for Training
You must split your data into train, validation, and test sets so the model can be evaluated properly.

The Splitting Script
Create a file named split_data.py in your dataset folder and paste this code:

Python

import os
import random
import shutil

# --- Configuration ---
source_folder = '.'  # Current folder
split_ratio = (0.8, 0.1, 0.1) # 80% train, 10% val, 10% test
# ---------------------

def split_dataset(source):
    all_files = [os.path.splitext(f)[0] for f in os.listdir(source) if f.endswith('.jpg')]
    random.shuffle(all_files)

    total_files = len(all_files)
    train_end = int(total_files * split_ratio[0])
    val_end = train_end + int(total_files * split_ratio[1])

    sets = {'train': all_files[:train_end], 'val': all_files[train_end:val_end], 'test': all_files[val_end:]}

    for set_name, file_list in sets.items():
        img_path = os.path.join(source, 'images', set_name)
        lbl_path = os.path.join(source, 'labels', set_name)
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(lbl_path, exist_ok=True)
        
        for base_name in file_list:
            shutil.move(os.path.join(source, f"{base_name}.jpg"), os.path.join(img_path, f"{base_name}.jpg"))
            shutil.move(os.path.join(source, f"{base_name}.txt"), os.path.join(lbl_path, f"{base_name}.txt"))
        print(f"Moved {len(file_list)} image/label pairs to {set_name} set.")

split_dataset(source_folder)
How to Use
Run python split_data.py in your dataset folder (which now contains .jpg and .txt files).

The script will automatically create the required folder structure and move the files.

Final Folder Structure
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
5. Step 4: Creating the data.yaml File
YOLO needs a .yaml file to understand your dataset's structure and class names.

Action: In your main dataset folder, create a file named data.yaml.

Content: Paste the following into the file, editing the path and names to match your project.

YAML

# Full path to your main dataset folder
path: /path/to/your/dataset  # IMPORTANT: Use absolute path for local training or update later for Colab
train: images/train
val: images/val
test: images/test

# Class names
names:
  0: MyClassName1
  1: MyClassName2
6. Step 5: Training the Model in Google Colab
Google Colab provides free GPU access, which is perfect for training.

Zip your dataset folder (containing images, labels, and data.yaml).

Upload the .zip file to your Google Drive.

Open a new Colab notebook and run the following cells.

Python

# 1. Install YOLOv8
!pip install ultralytics

# 2. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Unzip your dataset
# IMPORTANT: Update this path to where your zip file is in your Drive
!unzip /content/drive/MyDrive/dataset.zip -d /content/

# 4. IMPORTANT: Update the YAML path for Colab
import yaml
with open('/content/dataset/data.yaml', 'r') as f:
    data = yaml.safe_load(f)
data['path'] = '/content/dataset'
with open('/content/dataset/data.yaml', 'w') as f:
    yaml.dump(data, f)

# 5. Start Training
# This will use the small (n) pretrained model and train for 50 epochs
!yolo task=detect mode=train model=yolov8n.pt data=/content/dataset/data.yaml epochs=50 imgsz=640
⚠️ Common Problems & Solutions
Problem: The !unzip command fails.

Cause: The path to your .zip file in Google Drive is incorrect.

Solution: In the Colab file explorer, find your .zip file, right-click, and select "Copy path". Paste this correct path into the command.

Problem: Training fails with an error about not finding image paths.

Cause: You forgot to update the path variable inside your data.yaml file to the Colab path (/content/dataset).

Solution: Run step #4 from the Colab code above. This automatically fixes the YAML file for the Colab environment.

7. Step 6: Real-Time Detection with VS Code
Once training is complete, download the best.pt file from the /content/runs/detect/train/weights/ folder in Colab.

The Detection Script
Create a new project folder on your PC, place best.pt inside it, and create a run_detector.py file with this code:

Python

import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('best.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Run detection on the frame
    results = model(frame)

    # Get the annotated frame with boxes and labels
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("Real-Time Detection", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
How to Use
Install the required libraries: pip install opencv-python ultralytics.

Run the script from your terminal: python run_detector.py.

⚠️ Common Problems & Solutions
Problem: The script crashes with a ModuleNotFoundError for cv2 or ultralytics.

Cause: The libraries are not installed in the Python environment that VS Code is currently using.

Solution: In VS Code, press Ctrl+Shift+P, search for Python: Select Interpreter, and choose the Python installation where you ran the pip install command. The error indicators (squiggly lines) under your imports should disappear.


important this in cmd if any issue   and open cmd in that folder 
 ren convert.py.txt convert.py
python convert.py
del *.xml
