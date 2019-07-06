from typing import Dict, Any, Sequence, Tuple, List
import sys
from sqlalchemy import create_engine

import pandas as pd

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download(['punkt', 'wordnet','stopwords'])

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.ensemble import GradientBoostingClassifier

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
    engine.execute("SELECT * FROM disturbing_tweets").fetchall()
    df = pd.read_sql_table(table_name, engine)
    X = df.loc[:, 'message']
    y = df.iloc[:, 4:]
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
    for tok in tokenized:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model() -> GridSearchCV:
    '''
    Builds mdoel pipeline
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
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', GradientBoostingClassifier())
    ])

    parameters = {
        # 'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        # 'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        # 'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        # 'features__text_pipeline__tfidf__use_idf': (True, False),
        # 'clf__n_estimators': [50, 100, 200],
        # 'clf__min_samples_split': [2, 3, 4],
        # 'features__transformer_weights': (
        #     {'text_pipeline': 1, 'starting_verb': 0.5},
        #     {'text_pipeline': 0.5, 'starting_verb': 1},
        #     {'text_pipeline': 0.8, 'starting_verb': 1},
        # )
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
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

    y_pred = model.predict(X_test)
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_)

    for idx, col in enumerate(category_names):
        print(col, classification_report(Y_test.iloc[:,idx], y_pred[:,idx]))


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
