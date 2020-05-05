"""
@author yy
play with data
"""
import pickle

with open("../data/preprocessed_data/chosen_word2id.pkl", "rb") as f:
    chosen_word2id = pickle.load(f)

with open("../data/preprocessed_data/chosen_word_list.pkl", "rb") as f:
    word_list = pickle.load(f)

with open("../data/preprocessed_data/qa_pair.pkl", "rb") as f:
    qa_pairs = pickle.load(f)
