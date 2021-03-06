# Disaster Response Pipeline Project

### Table of Contents
1. Project Motivation
2. File Descriptions
3. Installation
4. Results
5. Licensing and Acknowledgements

### Project Motivation:
Disasters are terrible occurrences and we try to prevent and reduce their impacts where possible. This project makes use of data science and machine learning to classify messages sent during a disaster in order to allow first responders access the situation properly and send in appropriate teams where necessary.

### File Descriptions:
The project contains the following folders:
- app
  1. template : This contains the master.html and go.html pages that show the main page and classification report page of the web app respectively.

  2. run.py : The flask file that runs the app.

- data
  1. disaster_categories.csv : data file to process
  2. disaster_messages.csv  : data file to process
  3. process_data.py : python script that cleans the data and loads to a database.


- models
  1. train_classifier.py : python script that builds the machine learning model.



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
        A database is then created and saved in the "data" folder.
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        The trained ML model is created and saved in the "models" folder.

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to localhost:3001


### Results
The model performs well with an accuracy of 94%. The web page displays an input bar for the user to type in a message for classification. The web page also displays the distribution of message genres and a distribution of message categories used to train the model.
There are some categories with few examples and as such their Recall and F1-score are ill defined and set to zero(0). It is possible that some of these samples do not make the test split and as such are not predicted.  

- Screenshot of Web app
   <img width="1440" alt="Screen Shot 2021-11-02 at 11 20 15 AM" src="https://user-images.githubusercontent.com/87318106/139829146-97825e22-6650-48d1-9285-753649afe0c8.png">

### Licensing and Acknowledgements
All files in this repository are free to use.  
