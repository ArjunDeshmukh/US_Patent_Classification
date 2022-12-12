# US_Patent_Classification
This repository contains code for a web application which can find the "CPC" code for a US Patent based on the title/abstract

Training Data Used:
60% of US Patents from years 2001 to 2018

## Python version
This code was tested using Python 3.10.5

## Setup
First install and enable Kaggle API with the help of this page: https://github.com/Kaggle/kaggle-api#download-dataset-files
After downloading the repository, run the following commands from the downloaded repository folder:
```
mkdir Trained_Models
cd Trained_Models
kaggle datasets download arjunrdeshmukh/patclassmodels-cpu --unzip
cd ..
```
This wil download the trained model from kaggle public dataset: https://www.kaggle.com/datasets/arjunrdeshmukh/patclassmodels-cpu

## Running the application locally
You can run this application in your PC with the following commands:
````
py -m venv env
.\env\Scripts\activate
py -m pip install -r requirements.txt
flask_app.py
````

After running, flask_app.py you will get an output containing a line as follows:
Running on http://127.0.0.1:5000 (Press CTRL+C to quit)

Copy this URL from your command window and paste in your browser and press enter. This will show you a window where you can enter a patent abstract.
For more details about this project, you can read these blogs I wrote: 
Part 1: https://medium.com/@darjun94/us-patent-classification-end-to-end-nlp-project-part-1-51604a754bbe
Part 2: https://medium.com/@darjun94/us-patent-classification-end-to-end-nlp-project-part-2-d6538149f353
Part 3: https://medium.com/@darjun94/us-patent-classification-end-to-end-nlp-project-part-3-a8a6549881a9
