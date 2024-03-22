from sklearn.feature_extraction.text import TfidfVectorizer
from utils import constants
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import os
from tqdm import tqdm

ngram_rangetuple = (1, 1)

def extract_feature(feature, train_df, test_df, max_rows=1000):
    train_df, test_df = train_df[: max_rows], test_df[:max_rows]
    train_text_df, test_text_df = train_df.drop(constants.ColumnLabel, axis=1), test_df.drop(constants.ColumnLabel, axis=1)
    if feature == constants.FeatureTFIDF:
        train_features_df, test_features_df = generate_tf_idf_features(train_text_df, test_text_df)
    elif feature == constants.FeatureCountVectorizer:
        train_features_df, test_features_df = generate_count_vectorizer_features(train_df, test_df)
    elif feature == constants.FeatureGloVe:
        train_features_df, test_features_df = get_glove_word2vec_embedding(train_df, test_df)
    return pd.concat([train_features_df, train_df[constants.ColumnLabel]], axis=1), pd.concat([test_features_df, train_df[constants.ColumnLabel]], axis=1)

def generate_tf_idf_features(train_df, test_df):
    vectorizer = TfidfVectorizer(analyzer='word' , stop_words='english')
    train_features_df = pd.DataFrame(vectorizer.fit_transform(train_df[constants.ColumnData]).toarray())
    test_features_df = pd.DataFrame(vectorizer.transform(test_df[constants.ColumnData]).toarray())
    return train_features_df, test_features_df

def generate_count_vectorizer_features(train_df, test_df):
    vectorizer = CountVectorizer(ngram_range=ngram_rangetuple , stop_words='english')
    train_features_df = pd.DataFrame(vectorizer.fit_transform(train_df[constants.ColumnData]).toarray())
    test_features_df = pd.DataFrame(vectorizer.transform(test_df[constants.ColumnData]).toarray())
    return train_features_df, test_features_df

def get_glove_word2vec_embedding(train_df, test_df):
    vectorizer = GloveEmbeddingVectorizer()
    train_features_df = vectorizer.fit_transform(train_df[constants.ColumnData])
    test_features_df = vectorizer.transform(test_df[constants.ColumnData])
    return train_features_df, test_features_df


class GloveEmbeddingVectorizer:
    def __init__(self):
        glove_file = datapath(os.path.join(os.getcwd(), 'Data/Glove/glove.6B/glove.6B.100d.txt'))
        word2vec_glove_file = get_tmpfile("word2vec.txt")
        glove2word2vec(glove_file, word2vec_glove_file)

        self.glove_embedding_model = KeyedVectors.load_word2vec_format(word2vec_glove_file, binary=False)
        self.dims = self.glove_embedding_model.get_vector('business').shape[0]
    
    def transform(self, data):
        preprocessed_data = np.zeros((len(data), self.dims))
        n = 0
        empty_sentences = 0
        index = []
        for id, sentence in tqdm(data.items()):
            word_vecs = []
            if sentence:
                m = 0
                try:
                    words = sentence.split()
                except:
                    print(sentence, sentence == np.NaN)
                    continue
                for word in words:
                    try:
                        word_vector = self.glove_embedding_model.get_vector(word)
                        word_vecs.append(word_vector)
                        m += 1
                    except KeyError:
                        pass
            if len(word_vecs) > 0:
                word_vecs = np.array(word_vecs)
                preprocessed_data[n] = word_vecs.mean(axis=0)
                index.append(id)
            else:
                empty_sentences += 1
        if empty_sentences != 0:
            print("Number of sentences skipped:", empty_sentences)
            n += 1
        return pd.DataFrame(preprocessed_data, index=data.index)

    def fit_transform(self, data):
        return self.transform(data)