# Model File Requirements
## Libraries
Install the following:

    pip install numpy
    pip install matplotlib
    pip install torch
    pip install requests
    pip install json
    pip install beautifulsoup4

## Usage
At the top of the "main" method, you will find a variable, "load". This variable is defaulted to false, as to train the model. 
If you have a .pth stored in the directory "./models" then you may set that variable to true to use that model.
Simply run the file as follows:

    python final_predictor.py

The file will then train and use the model to predict the next 7 days of new cases and generate the .csv and .png





