import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'models/intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'models/tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'models/tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'models/word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    embeddings = dict()
    embeddings_dim = 100

    with open(embeddings_path, 'r') as ss_tsv:
        for line in ss_tsv:
            key = line.split('\t')[0]
            value = np.array(line.strip().split('\t')[1:], dtype=np.float32)

            if len(value) != embeddings_dim:
                continue

            embeddings[key] = value

    return embeddings, embeddings_dim


def question_to_vec(question, embeddings, dim=300):
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    question_embeddings = np.zeros(dim, dtype=np.float32)
    n_words = 0
    for idx, word in enumerate(question.split(' ')):
        if word in embeddings:
            n_words += 1
            question_embeddings += embeddings[word]

    return question_embeddings if n_words == 0 else question_embeddings / n_words


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
