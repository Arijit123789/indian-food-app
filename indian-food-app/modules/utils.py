import pandas as pd

def clean_text_column(df, column):
    # Lowercase and strip whitespace
    df[column] = df[column].str.lower().str.strip()
    return df

def fill_missing_values(df, fill_value='-'):
    # Fill NA/NaN values in all columns with a default value
    df = df.fillna(fill_value)
    return df

def encode_column(df, column):
    # Label encoding for categorical columns
    df[column] = pd.factorize(df[column])[0]
    return df

def get_unique_values(df, column):
    # Return sorted list of unique values in a column
    return sorted(df[column].unique())

def get_state_distribution(df):
    # Returns value counts for states
    return df['state'].value_counts()

def simplify_ingredients(ingredient_str):
    # Splits and cleans a string of ingredients into a list
    return [ingredient.strip().lower() for ingredient in ingredient_str.split(',')]

# Add more helper functions here as needed!
