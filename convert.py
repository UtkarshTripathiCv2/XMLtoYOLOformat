import glob
import os
import xml.etree.ElementTree as ET

# IMPORTANT: Edit this list to match the order in your classes.txt file
# Since your project is person-specific, you likely only have one class.
classes = ["JacKALKI/Utkarsh"]

# This function converts the PASCAL VOC coordinates to YOLO format
def convert_voc_to_yolo(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

# This function processes all XML files in the current directory
def convert_annotations():
    # Loop through all .xml files in the current folder
    for xml_file in glob.glob("*.xml"):
        
        # Create a matching .txt file path
        txt_file_path = os.path.splitext(xml_file)[0] + ".txt"
        
        # Open the .txt file for writing
        with open(txt_file_path, "w") as out_file:
            # Parse the XML file
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image size
            size = root.find("size")
            w = int(size.find("width").text)
            h = int(size.find("height").text)

            # Find all 'object' elements in the XML
            for obj in root.iter("object"):
                cls = obj.find("name").text
                # If the class is not in our list, skip it
                if cls not in classes:
                    continue
                
                # Get the class ID (index in the list)
                cls_id = classes.index(cls)
                
                # Get the bounding box
                xmlbox = obj.find("bndbox")
                b = (
                    float(xmlbox.find("xmin").text),
                    float(xmlbox.find("xmax").text),
                    float(xmlbox.find("ymin").text),
                    float(xmlbox.find("ymax").text),
                )
                
                # Convert coordinates and write to the .txt file
                bb = convert_voc_to_yolo((w, h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n")
                
    print("Conversion complete! All .xml files have been converted to .txt.")

# Run the conversion
if __name__ == "__main__":
    convert_annotations()