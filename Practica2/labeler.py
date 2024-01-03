import os
import xml.etree.ElementTree as ET

def parse_xml_and_create_txt(xml_file, output_folder):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract information from the XML
    filename = root.find('filename').text
    name = root.find('.//name').text
    xmin = int(root.find('.//xmin').text)
    ymin = int(root.find('.//ymin').text)
    xmax = int(root.find('.//xmax').text)
    ymax = int(root.find('.//ymax').text)

    # Determine the label based on the first character of the filename
    label = 0 if filename[0].isupper() else 1

    # Create the text content
    content = f"{label} {xmin} {ymin} {xmax} {ymax}"

    # Create a txt file with the same name as the filename (without extension)
    txt_filename = os.path.splitext(filename)[0] + '.txt'
    
    # Save the txt file in the 'labels' folder
    txt_path = os.path.join(output_folder, txt_filename)
    with open(txt_path, 'w') as txt_file:
        txt_file.write(content)

if __name__ == "__main__":
    # Specify the directory containing XML files
    xml_folder = "Practica2/data/old_pets/annotations/xmls"
    # './data/old_pets/annotations/xmls/'
    
    # Specify the output folder for text files
    output_folder = 'Practica2/data/pets/Yolo/train_val'

    # Create the 'labels' folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through XML files in the specified folder
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_folder, xml_file)
            parse_xml_and_create_txt(xml_path, output_folder)
