# Graph Trace Extractor
==========================

Table of Contents
-----------------

1. [Introduction](#introduction)
2. [Features](#features)
3. [User Guide](#user-guide)
4. [Development](#development)
5. [Troubleshooting](#troubleshooting)

## Introduction
The Graph Trace Extractor is a Python-based application designed to extract traces from graph images.
![Title Selection](assest/GTE_workflow.png)

## Features
* Extracts traces from graph images
* Detects grid lines and extracts numeric values
* Reconstructs the original trace
* Saves the extracted trace as a CSV file
* Displays the extracted trace as a plot

## User Guide
### Running the Application
To use the Graph Trace Extractor, follow these steps:

1. **Download the executable file or build the application from source. Application Download: [GTE.exe](dist/GTE.exe)**
2. Run the application and select a folder containing graph images.
3. The application will process the images and extract the traces.
4. The extracted traces will be displayed as plots in the application window.
5. The extracted traces will also be saved as CSV files in the selected folder.


### Application Images
The Graph Trace Extractor has a simple and intuitive interface. Here are some screenshots of the application:

#### Main Window
![Main Window](assest/open_app.png)

The main window of the application displays a list of available actions, including selecting a folder and processing images.

#### Folder Selection
![Folder Selection](assest/select_folder.png)

To select a folder, click on the "Select Folder" button and choose the folder containing your graph images.

#### Image Processing
![Image Processing](assest/processing_graph.png)

The application will process the images and extract the traces. This may take a few seconds, depending on the number of images and their complexity.

#### Plot Display
![Plot Display](assest/processed.png)

The extracted traces will be displayed as plots in the application window. You can zoom in and out, and pan the plot to view the data in more detail.

#### CSV Export
![CSV Export](assest/save_csv.png)

The extracted traces will also be saved as CSV files in the selected folder. You can open these files in any spreadsheet software to view and analyze the data.


## Development
### Setting up the Python Environment
To develop the Graph Trace Extractor, you will need to set up a Python environment with the required packages. Here are the steps to follow:

1. **Install Python**: Download and install Python 3.8 or later from the official Python website.
2. **Install pip**: pip is the package installer for Python. It is included with Python, so you don't need to install it separately.
3. **Create a new virtual environment**: To keep your project's dependencies separate from your system's Python environment, create a new virtual environment using the following command:
```bash
python -m venv gte-env
```
4. **Activate the virtual environment**: To activate the virtual environment, use the following command:
```bash
gte-env\Scripts\activate
```
On macOS or Linux, use the following command:
```bash
source gte-env/bin/activate
```
5. **Install required packages**: Install the required packages using pip:
```bash
pip install -r requirements.txt --proxy "http://webproxy.ext.ti.com:80"
```

6. **Set up Tesseract-OCR**

To set up Tesseract-OCR, follow these steps:

* Locate the `Tesseract-OCR` folder in the root directory, which contains the `tesseract.exe` file.
* Configure the `pytesseract` library to use the `tesseract.exe` file by setting the `tesseract_cmd` variable:
```python
pytesseract.pytesseract.tesseract_cmd = resource_path('Tesseract-OCR//tesseract.exe')
```
* When building the `.exe` file, use the following configuration:
```python
pytesseract.pytesseract.tesseract_cmd = resource_path('tesseract.exe')
```
Note that the path to the `tesseract.exe` file may vary depending on your system configuration. Make sure to update the `resource_path` function accordingly to point to the correct location of the `tesseract.exe` file.
    ```
7. **Build the application**: Build the application using PyInstaller:
```bash
pyinstaller --onefile --windowed --add-data "Tesseract-OCR;." GTE.py
```
This will create a standalone executable file in dist folder as `GTE.exe` that can be run on any Windows system without requiring Python or any other dependencies to be installed.

### Running without GUI
To run the application without the GUI, use the following command:
```python
from graph_processor import process_entire_image_folder
process_entire_image_folder('test')
```
Replace `'test'` with the path to the folder containing your graph images. For the testng, we have some graph in `test` folder.

## Troubleshooting
If you encounter any issues while using the Graph Trace Extractor, try the following:

* Check the log output for any error messages
* Verify that the required packages are installed correctly
* Try running the application in debug mode to get more detailed error messages
