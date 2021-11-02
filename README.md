# Disaster Response Pipeline Project

### Table of Contents
Project Motivation
File Descriptions
Installation
Results
Licensing and Acknowledgements

### Project Motivation:
Disasters are terrible occurrences and we try to prevent and reduce their impacts where possible. This project makes use of data science and machine learning to classify messages sent during a disaster in order to allow first responders properly access the situation and send in appropriate teams where necessary.

### File Descriptions:
The project contains the following folders:
- app
  template : This contains the master.html and go.html pages that show the main page and classification report page of the web app respectively.

  run.py : The flask file that runs the app.

- data
  disaster_categories.csv : data file to process
  disaster_messages.csv  : data file to process
  process_data.py : python script that cleans the data and loads to a database.
  InsertDatabaseName.db : database holding cleaned data.

- models
  train_classifier.py : python script that builds the machine learning model.
  classifier.pkl : The saved model


### Installation:
The following libraries are needed to make use of the code
 - numpy
 - pandas
 - sqlalchemy
 - sklearn
 - nlkt
 - plotly
 - joblib
 - flask


1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to localhost:3001

### Results
The model performs well with an accuracy of 94%. The web page displays an input bar for the user to type in a message for classification. The web page also displays the distribution of message genres and a distribution of message categories used to train the model.
There are some categories with few examples and as such their Recall and F1-score are ill defined and set to zero(0). It is possible that some of these samples do not make the test split and as such are not predicted.  

### Licensing and Acknowledgements
All files in this repository are free to use.  
