# Information Extractor from Production Label

## What is it?
This project is designed to extracting information (text) from production labels which would help to reduce the working time of grabbing it by hands.
### Example:
From input label:
![alt text](https://github.com/thuyluu9595/textExtraction/blob/master/ImageTemplate/image0.jpg?raw=true)
CSV file output:

<img width="865" alt="Screen Shot 2022-10-17 at 8 02 13 AM" src="https://user-images.githubusercontent.com/78382696/196213954-75230401-2fe2-47ff-b95d-ecbf0a0cbff1.png">

## Main Features
Here are some features of this program:
    
    - Accept multiple images of labels as input.
    - Output is a .csv file which can be easily processed or converted to Excel file to process the data.
    - User can choose the text area of interest inn the label and name of categories for the output file.

## Installation and Usage
 1. Install Python at: https://www.python.org/downloads/. Install pip at: https://pip.pypa.io/en/stable/installation/
 2. Download the zip file or clone the project with Git by using the command:
    ```sh
    git clone https://github.com/thuyluu9595/textExtraction.git
    ```
 3. Unzip if the file if downloading the zip file. In the project directory, run the command to install dependencies:
    ```sh
    # Python3
    pip3 install -r requirements.txt
    ```
    or
    ```sh
    # Python2
    pip install -r requirements.txt
    ```
 4. Place all the images of labels for extracting information in the **InputImages** folder. Place a template image in the **ImageTemplate** folder.
 5. Run the program:
    
    - If using the default label form and categories, run the command:
    ```sh
    python main.py
    ```
    - If using a customized label, use the command:
    ```sh
    python main.py ROI
    ```
    then choose the area of interest by selecting the top left and right bottom point and type in the name of category. When finished, press 's' key.
 6. After processing, the output file is located in the **Data** folder.
