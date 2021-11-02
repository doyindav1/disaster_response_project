# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges the source files
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge the datasets
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """
    cleans the data and prepares the data for ML pipeline
    """

    # split the values in the categories column on the ; character
    new_category = df['categories'].str.split(";", expand=True)

    # rename column of categories with new column names
    row = new_category.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    new_category.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in new_category:
       # set each value to be the last character of the string
        new_category[column] = pd.Series(new_category[column]).astype(str)
        new_category[column] = new_category[column].apply(lambda x: x.split("-")[1])
        # convert string to numeric
        new_category[column] = pd.to_numeric(new_category[column])

    # replace categories column in original df with new category columns
    dropped_df = df.drop(columns=['categories'])
    concat_df = pd.concat([dropped_df, new_category], axis=1)
    concat_df.related.replace(2,1,inplace=True)

    # remove duplicates
    cleaned_df = concat_df.drop_duplicates()

    return cleaned_df


def save_data(cleaned_df, database_filename):
    """
    saves cleaned dataframe to a local sql database
    """
    engine = create_engine('sqlite:///' + str(database_filename))
    cleaned_df.to_sql('disaster_messages', engine, index=False, if_exists='replace')

    return


def main():
    """
    Runs functions above
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
