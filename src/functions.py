"""Your reusable functions live here.

The rule from the brief: Python logic goes in `.py` files, and the notebooks
hold the narrative. The moment a cell grows past a few lines, or you find
yourself pasting it a second time, move it here and call it from the notebook.

To use this module from a notebook in `notebooks/`:

    import sys
    sys.path.append("..")
    from src.functions import *

The four below are only there to show the shape. Rename them, change the
arguments, write your own. This is a starting point, not an interface you have
to implement.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"

RANDOM_STATE = 42


def load_data():
    """Read your raw file and apply the fixes that do not learn from the data."""
    pass


def build_preprocessor(numeric, categorical):
    """Return a ColumnTransformer that prepares each type of column."""
    pass


def compare_models(models, X, y, cv, scoring):
    """Cross-validate each model on the same folds and return a tidy table."""
    pass


def plot_errors(y_true, y_pred):
    """Show where the model goes wrong: a confusion matrix or the residuals."""
    pass


def convert_binary_columns(df, column_groups):
    """Converts a group a binary columns in one categorical column
         Input:
        - df: pandas dataframe
        - column_groups: dictionnary where each key is the name of a categorical column
                            and each values is a list of the binary columns categrories
                            
        Returns: a new pandas dataframe with the new categorical columns
                     the binary columns are removed
            
        Note: This function assumes that each binary column name ends with a
                    numeric category identifier, which is extracted to represent the
                    corresponding categorical class."""
    
    for new_column, binary_columns in column_groups.items():

        # Identifing the active category
        df[new_column] = (
            df[binary_columns]
            .idxmax(axis=1)
            .str.extract(r"(\d+)$")[0]
            .astype(int))

        # Remove the original binary columns
        df.drop(columns=binary_columns, inplace=True)

    return df


def column_names_std(df):
    """Standardises column names using snake_case
        - removing leading and trailing spces
        - setting everything to lower case
        - replacing every character that is not a letter or a number with an underscore
        Input: the dataframe
        Output: dataframe with standardised column names"""
    df = df.copy()
    df.columns = (df.columns
        .str.strip()# removing spaces
        .str.lower()# everything in lower case
        .str.replace(r"[^a-z0-9]+", "_", regex=True)# replacing every character that is not a letter or a number with an underscore 
        )
    return df


def split (features, target, size, random_state,n_splits ):
    """Splits the data into training and test sets using stratification on the target variable. 
        Ensures reproducible splits and allows the same cross-validation folds to be reused across different notebooks.
     Input: 
      - features: the input features (X)
      - target: the target variable (y)
      - size: the proportion of data reserved for the test set
      - random_state: the random seed chosen to ensure reproducibility
      - n_spits: the number of folds used for cross validation
     Output:
      - prints the number of rows in the training and testing data
      - returns X, y, X_train, X_test, y_train, y_test, cv, folds """
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold
    X = features
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= size, stratify= y, random_state= random_state)
    cv = StratifiedKFold(n_splits= n_splits, shuffle=True, random_state= random_state)
    folds = list(cv.split(X_train, y_train))
    print(len(X_train), "training rows,", len(X_test), "test rows")
    return X, y, X_train, X_test, y_train, y_test, cv, folds