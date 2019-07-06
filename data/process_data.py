import sys
import pandas as pd
import sqlite3

def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    '''
    Loads messages and categories and merge them together
    so result contains category labels for each message.

    Parameters
    ----------
    messages_filepath : str
        messages file path
    categories_filepath : str
        categories file path

    Returns
    -------
    pd.DataFrame
        with the following columns
        - id, message, original, genre, categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on=['id'])

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Removes the records with no data.

    Parameters
    ----------
    df : pd.DataFrame
        cleaned data
    Returns
    -------
    pd.DataFrame
        that contain only full records
    '''
    categories = df['categories'].str.split(n = 36, pat = ';', expand = True)

    # cleanup column name
    row = categories.iloc[0,:]
    category_colnames = [val.split('-')[0] for val in row.values]
    categories.columns = category_colnames

    # encode the column
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        categories[column] = pd.to_numeric(categories[column])

    # replace with new categories
    df.drop('categories', axis= 1, inplace =True)
    df = pd.concat([df, categories], axis=1, join_axes=[df.index])
    return df.drop_duplicates()


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    '''
    Saves dataframe to sqlite database so results contain.

    Parameters
    ----------
    df : pd.DataFrame
        records to save
    database_filename : str
        database file name
    '''
    with sqlite3.connect(database_filename) as conn:
        df.to_sql("disturbing_tweets", conn, if_exists='replace')
        conn.commit()

def main() -> None:
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
