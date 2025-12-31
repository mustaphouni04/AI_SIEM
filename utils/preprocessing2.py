"""
Use this file to get the necessary data to train a model.
"""
# Importing all necessary preprocessing functions from utils
import pandas as pd
import utils2
from utils2 import preprocessing_TS, preprocessing_SC, preprocessing_UA, preprocessing_R, preprocessing_RU, preprocessing_RM, preprocessing_RH, preprocessing_BS, preprocessing_CLEAN, after_preprocessing

# Reload utils to update any changes made to the script
import importlib
importlib.reload(utils2)


def final_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    args --> The original DataFrame with the log files.
    output --> The DataFrame with all important and numerical features preprocessed.
    """
    # Create copies because of SQL unsupported type when getting lists (embeddings)
    df_copy = df.copy()
    df_copy2 = df.copy()
    df_copy3 = df.copy()
    
    # Numerical features, one hot, normalize...
    print("Dropping all unnecessary features...")
    df = preprocessing_CLEAN(df)
    print("Done.")
    print("Preprocessing timestamp...")
    df = preprocessing_TS(df)
    print("Done.")
    print("Preprocessing status_code...")
    df = preprocessing_SC(df)
    print("Done.")
    print("Preprocessing request_method...")
    df = preprocessing_RM(df)
    print("Done.")
    print("Preprocessing bytes_sent...")
    df = preprocessing_BS(df)
    print("Done.")

    # Using the copies to get the correspondant embeddings for textual features
    print("Preprocessing referer...")
    df_copy = preprocessing_R(df_copy)
    referer_column = df_copy['referer']
    df['referer'] = referer_column
    print("Done.")
    print("Preprocessing requested_url...")
    df_copy2 = preprocessing_RU(df_copy2)
    requested_url_column = df_copy2['requested_url']
    df['requested_url'] = requested_url_column
    print("Done.")
    print("Preprocessing user_agent...")
    df_copy3 = preprocessing_UA(df_copy3)
    user_agent_column = df_copy3['user_agent']
    df['user_agent'] = user_agent_column
    print("Done.")

    print("Preprocessing referer, user_agent and requested_url...")
    df = after_preprocessing(df)
    print("Done.")
    
    return df
