# import libraries
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """
    loads the data from the SQL db and produces X and Y values for ML pipeline
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages', engine)
    X = df['message'].values
    Y = df.drop(columns=['id', 'message', 'original', 'genre']).values
    category_names = df.drop(columns=['id', 'message', 'original', 'genre']).columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """
    tokenizes the text
    """
        # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    ML model built using a pipeline and GridSearchCV
    """
    pipeline = Pipeline([

        ('vect', CountVectorizer(tokenizer=tokenize)),

        ('tfidf', TfidfTransformer()),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))

    ])

    parameters = {

                'clf__estimator__n_estimators': [50, 100, 200],
                'clf__estimator__min_samples_split': [2, 4]

                  }

    model = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=2, verbose=3)


    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This applies the model on the data and prints the classification report for each category_names
    """

    Y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print("category: ", category)
        print(classification_report(Y_test[i],Y_pred[i]))

    accuracy = (Y_pred == Y_test).mean()
    print("Accuracy:", accuracy)

    return


def save_model(model, model_filepath):
    """
    This saves the model as a pickle file
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))

    return


def main():
    """
    This function loads the data, splits the data into train and test sets,
    trains and saves the model using the functions in the script
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
