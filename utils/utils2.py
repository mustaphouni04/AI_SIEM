# IMPORTING NECESSARY LIBRARIES
#_________________________________________________________________________________________________________
import pandas as pd
import numpy as np
import nltk
import re
import sqlite3
import matplotlib.pyplot as plt
from string import punctuation
#from keras.preprocessing import text
from urllib.parse import unquote
from nltk.tokenize import word_tokenize
from nltk.corpus import PlaintextCorpusReader
from gensim.models import Word2Vec
import multiprocessing
import os
from sklearn.preprocessing import MinMaxScaler
import hashlib
import ast

# HELPER FUNCTIONS
#_________________________________________________________________________________________________________
def preprocess_useragent(user_agent):
    user_agent = re.sub(r'[/.\-(),;:]', ' ', user_agent)
    tokens = word_tokenize(user_agent)
    return tokens
    
def preprocess_referer(ref):
    ref = re.sub(r'[/.-:_=;,]', ' ', ref)
    ref = re.sub(r'-', ' ', ref)
    tokens = word_tokenize(ref)
    return tokens

def preprocess_url(url):
    url = re.sub(r'[/.-]', ' ', url)
    tokens = word_tokenize(url)
    return tokens
    
def generate_embedding_vectors(feature_tokens, word_vectors, emb_dim):
    embedding_vectors = []
    for token in feature_tokens:
        if token in word_vectors:
            embedding_vectors.append(word_vectors[token])
        else:
            embedding_vectors.append([0.0] * emb_dim)  # if token not found, fill with zeros
    return embedding_vectors
    
def embeddings(df: pd.DataFrame, preprocess, feature, emb_dim = 30, epochs = 10, window = 5) -> pd.DataFrame:
    db_file = f'generic_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.db'
    # load data
    conn = sqlite3.connect(db_file)
    df.to_sql('generic', conn, index=False, if_exists='replace')

    # tokenize feature and create corpus
    corpus_path = "corpus"
    gen_tokens = []

    chunk_size = 10000
    for chunk in pd.read_sql_query(f'SELECT {feature} FROM generic', conn, chunksize=chunk_size):
        feats = chunk[feature]
        for feat in feats:
            decoded_feat = unquote(feat)
            tokens = preprocess(decoded_feat)
            gen_tokens.append(tokens)

    conn.close()

    os.remove(db_file)
    
    # getting the embeddings
    EMB_DIM = emb_dim
    w2v = Word2Vec(sentences=gen_tokens, vector_size=EMB_DIM, window=window, min_count=5, negative=15,
               epochs=epochs, workers=multiprocessing.cpu_count())

    word_vectors = w2v.wv
    
    # get the vocabulary
    vocab = list(word_vectors.key_to_index.keys())
    
    # get the word vectors
    word_vecs = [word_vectors[word] for word in vocab]
    
    # convert word_vecs to a numpy array
    word_vecs_np = np.array(word_vecs)

    # create a new list to hold the embedding vectors
    embedding_vectors_list = []

    # iterate over each observation in the dataframe
    for i, row in df.iterrows():
        row_feature = row[feature]
        decoded_feature = unquote(row_feature)
        tokens = preprocess(decoded_feature)
        embedding_vectors = generate_embedding_vectors(tokens, word_vectors, EMB_DIM)
        embedding_vectors_list.append(embedding_vectors)

    df.drop(feature, axis = 1, inplace = True)
    # add the list of embedding vectors as a new column to the dataframe
    df[feature] = embedding_vectors_list

    return df

# PREPROCESSING FUNCTIONS
#_________________________________________________________________________________________________________
def preprocessing_TS(df: pd.DataFrame) -> pd.DataFrame:
    """
    TS: Cyclic encoding of timestamp.
    """
    # convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%b/%Y:%H:%M:%S %z')

    # extract hour, minute, and second components
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second

    # encode hour, minute, and second with sine and cosine functions
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['second_sin'] = np.sin(2 * np.pi * df['second'] / 60)
    df['second_cos'] = np.cos(2 * np.pi * df['second'] / 60)

    # drop original timestamp and hour, minute, and second columns
    df.drop(['timestamp', 'hour', 'minute', 'second'], axis=1, inplace=True)

    return df

def preprocessing_SC(df: pd.DataFrame) -> pd.DataFrame:
    """
    SC: One-Hot encoding status_code.
    """
    one_hot_encoded = pd.get_dummies(df['status_code'], prefix='status_code')
    df_encoded = pd.concat([df, one_hot_encoded], axis=1)
    df_encoded.drop('status_code', axis=1, inplace=True)

    return df_encoded

def preprocessing_RM(df: pd.DataFrame) -> pd.DataFrame:
    """
    RM: One-Hot encoding request_method.
    """
    one_hot_encoded = pd.get_dummies(df['request_method'], prefix='request_method')
    df_encoded = pd.concat([df, one_hot_encoded], axis=1)
    df_encoded.drop('request_method', axis=1, inplace=True)

    return df_encoded

def preprocessing_RH(df: pd.DataFrame) -> pd.DataFrame:
    """
    RH: Normalizing remote_host and splitting into octets.
    """
    raise NotImplementedError()

def preprocessing_R(df: pd.DataFrame) -> pd.DataFrame:
    """
    R: Generating embeddings for referer.
    """
    
    df_with_embeddings = embeddings(df, preprocess_referer, feature = 'referer', emb_dim = 30, epochs = 10, window = 5)  

    return df_with_embeddings

def preprocessing_RU(df: pd.DataFrame) -> pd.DataFrame:
    """
    RU: Generating embeddings for requested_url.
    """   
    df_with_embeddings = embeddings(df, preprocess_url, feature = 'requested_url', emb_dim = 30, epochs = 10, window = 5)  

    return df_with_embeddings

def preprocessing_BS(df: pd.DataFrame) -> pd.DataFrame:
    """
    BS: Normalizing bytes_sent.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit the scaler to the data and transform the 'bytes_sent' column
    df['bytes_sent'] = scaler.fit_transform(df['bytes_sent'].values.reshape(-1, 1))
    
    return df
    
def preprocessing_UA(df: pd.DataFrame) -> pd.DataFrame:
    """
    UA: Generating embeddings for user_agent.
    """
    df_with_embeddings = embeddings(df, preprocess_useragent, feature = 'user_agent', emb_dim = 30, epochs = 10, window = 5)

    return df_with_embeddings

def preprocessing_CLEAN(df: pd.DataFrame) -> pd.DataFrame:
    """
    CLEAN: Drop all unnecessary columns.
    """
    df.drop('server_name', axis=1, inplace=True)
    df.drop('remote_logname', axis=1, inplace=True)
    df.drop('remote_user', axis=1, inplace=True)

    return df

def after_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # replacing [] observations by a 30 dim vector of zeroes
    zero_vector = [0.0] * 30
    df['referer'] = df['referer'].apply(lambda x: zero_vector if x == [] else x)
    df['user_agent'] = df['user_agent'].apply(lambda x: zero_vector if x == [] else x)
    df['requested_url'] = df['requested_url'].apply(lambda x: zero_vector if x == [] else x)

    # do the mean of all words in the embedding iff observation is not 30 dim zero vector
    def convert_to_vector(embedding):
        # here embedding is treated as a NumPy array for consistent handling
        embedding_array = np.asarray(embedding)
        
        if embedding_array.shape == (30,) and np.all(embedding_array == 0.0):
            # return  zero vector itself if it matches the condition
            return embedding_array.tolist()
        elif embedding_array.shape!= (30,) or not np.all(embedding_array == 0.0):
            # proceed with original logic if the embedding is not a zero vector
            mean_vector = np.mean(embedding_array, axis=0)
            return mean_vector.tolist()
        else:
            # handle other cases where the input does not match expected formats
            return None

    df['referer'] = df['referer'].apply(convert_to_vector)
    df['user_agent'] = df['user_agent'].apply(convert_to_vector)
    df['requested_url'] = df['requested_url'].apply(convert_to_vector)

    # get every value in the vectors as a single feature for each referer, user_agent or requested_url
    def split_vector_column(df, column_name):
        # create new columns for each element in the vector
        vector_cols = [f"{column_name}_{i}" for i in range(30)]
        vector_df = pd.DataFrame(df[column_name].tolist(), columns=vector_cols)
        
        # drop the original vector column and concatenate the new columns
        df = df.drop(columns=[column_name])
        df = pd.concat([df, vector_df], axis=1)
    
        return df
        
    df = split_vector_column(df, 'requested_url')
    df = split_vector_column(df, 'referer')
    df = split_vector_column(df, 'user_agent')

    return df





