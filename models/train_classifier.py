from typing import Dict, Any, Sequence, Tuple, List
import sys
from sqlalchemy import create_engine

import pandas as pd

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import pickle

nltk.download(['punkt', 'wordnet','stopwords'])

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
stop_words = stopwords.words("english")

def load_data(database_filepath: str, table_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, Sequence[str]]:
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
    engine = create_engine(f'sqlite:///{database_filepath}', echo=False)
    df = pd.read_sql_table(table_name, engine)
    X = df.loc[:, 'message']
    y = df.iloc[:, 5:]
    category_names = list(y.columns)
    return X, y, category_names

def tokenize(text: str) -> Sequence[str]:
    '''
    Applies a number of tokenizing related transformations
    - Tokenize text into words
    - Lemmatize the tokens into stem
    - resulting tokens lowercased and trimmed

    Parameters
    ----------
    text : str
        input message

    Returns
    -------
    Sequence[str]
        a list of tokens
    '''
    lemmatizer = WordNetLemmatizer()
    tokenized = word_tokenize(text)
    clean_tokens = []
    for token in tokenized:
        if token in stop_words:
            continue
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model() -> GridSearchCV:
    '''
    Builds model pipeline
    - Tokenize text into words
    - Lemmatize the tokens into stem
    - resulting tokens lowercased and trimmed

    Parameters
    ----------
    text : str
        input message

    Returns
    -------
    Sequence[str]
        a list of tokens
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state = 42)))
    ])


    parameters = {
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        # 'vect__max_features': (None, 5000, 10000),
        # 'tfidf__use_idf': (True, False),
        # 'clf__estimator__n_estimators': [10, 20, 50, 100, 200],
        'clf__estimator__n_estimators': [50],
        # 'clf__estimator__min_samples_split': [2, 3, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters) #, n_jobs=-1)
    return cv

def evaluate_model(model: GridSearchCV, X_test: pd.DataFrame, Y_test: pd.DataFrame, category_names: Sequence[str]) -> None:
    '''
    Evaluates the model and saves the report.

    Parameters
    ----------
    model : GridSearchCV
        grid search model
    X_test : pd.DataFrame
        dataframe containing test X
    Y_test : pd.DataFrame
        dataframe containing test Y
    category_names: Sequence[str]
        the list of category names
    '''

    Y_pred = model.predict(X_test)
    labels = np.unique(Y_pred)
    accuracy = (Y_pred == Y_test).mean()

    print("Labels: ", labels)
    print("Accuracy: ", accuracy)
    print("Best Parameters: ", model.best_params_)

    for i, name in enumerate(category_names):
        print(name, classification_report(Y_test.iloc[:,i], Y_pred[:,i]))


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    '''
    Saves the model.

    Parameters
    ----------
    model : Pipeline
        model
    model_filepath : str
        path to save the model
    '''
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath, "disturbing_tweets")
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
