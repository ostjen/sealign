import gensim
import re
from nltk.corpus import stopwords
import logging
from text_formatter import *
from gensim.corpora.dictionary import Dictionary
import numpy as np
from numpy import dot, zeros, dtype, float32 as REAL,\
    double, array, vstack, fromstring, sqrt, newaxis,\
    ndarray, sum as np_sum, prod, ascontiguousarray,\
    argmax
from pyemd import emd


def wmdistance_b(src_model, tgt_model, document1, document2):
    """
    .. Ofir Pele and Michael Werman, "A linear time histogram metric for improved SIFT matching".
    .. Ofir Pele and Michael Werman, "Fast and robust earth mover's distances".
    .. Matt Kusner et al. "From Word Embeddings To Document Distances".
    Note that if one of the documents have no words that exist in the
    Word2Vec vocab, `float('inf')` (i.e. infinity) will be returned.
    This method only works if `pyemd` is installed (can be installed via pip, but requires a C compiler).
   """

    logger = logging.getLogger(__name__)
    PYEMD_EXT = True
    if not PYEMD_EXT:
        raise ImportError("Please install pyemd Python package to compute WMD.")

    # Remove out-of-vocabulary words.
    len_pre_oov1 = len(document1)
    len_pre_oov2 = len(document2)
    diff1 = len_pre_oov1 - len(document1)
    diff2 = len_pre_oov2 - len(document2)
    if diff1 > 0 or diff2 > 0:
        logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)

    if len(document1) == 0 or len(document2) == 0:
        logger.info(
            "At least one of the documents had no words that werein the vocabulary. "
            "Aborting (returning inf)."
        )
        return float('inf')

    dictionary = Dictionary(documents=[document1, document2])
    vocab_len = len(dictionary)

    if vocab_len == 1:
        # Both documents are composed by a single unique token
        return 0.0

    # Sets for faster look-up.
    docset1 = set(document1)
    docset2 = set(document2)
    # Compute distance matrix.
    distance_matrix = zeros((vocab_len, vocab_len), dtype=double)
    for i, t1 in dictionary.items():
        for j, t2 in dictionary.items():
            if t1 not in docset1 or t2 not in docset2:
                continue
            # Compute Euclidean distance between word vectors.
            distance_matrix[i, j] = sqrt(np_sum((src_model[t1] - tgt_model[t2]) ** 2))

    if np_sum(distance_matrix) == 0.0:
        # `emd` gets stuck if the distance matrix contains only zeros.
        logger.info('The distance matrix is all zeros. Aborting (returning inf).')
        return float('inf')


    def nbow(document):
        d = zeros(vocab_len, dtype=double)
        nbow = dictionary.doc2bow(document)  # Word frequencies.
        doc_len = len(document)
        for idx, freq in nbow:
            d[idx] = freq / float(doc_len)  # Normalized word frequencies.
        return d


    # Compute nBOW representation of documents.
    d1 = nbow(document1)
    d2 = nbow(document2)

    # Compute WMD.
    return emd(d1, d2, distance_matrix)


def generate_matrix(src_model, tgt_model, src_sentences, tgt_sentences,csv = 'no'):
    """""generates a csv file comparing all the sentences distances using wmd method.

         representation:
             |d00 d01| -> d00 = distance beetween src_sentence[0] and tgt_sentence[0]
             |d10 d11|    d01 = distance beetween src_sentence[0] and tgt_sentence[1]
                          d10 = distance beetwen src_sentence [1] and tgt_sentence[0]
                          d11 = distance beetwen src_sentence [1] and tgt_sentence[1]

    """
    matrix = np.zeros((len(src_sentences), len(tgt_sentences)))
    for i in range(0, len(src_sentences)):
        distances = []
        for j in range(0, len(tgt_sentences)):
            distances.append(wmdistance_b(src_model, tgt_model, src_sentences[i], tgt_sentences[j]))
        matrix = np.insert(matrix, i, distances, 0)
    matrix = matrix[:len(src_sentences)]
    if csv == 'yes':
        np.savetxt("result_matrix4.csv",matrix, delimiter=",")
    return matrix

def matrix_evaluation(matrix):
    result = 0
    for i in range(0,matrix.shape[0]):
        if min(matrix[i]) == matrix[i][i]:
            result = result + 1
    return result/matrix.shape[0]


def load_care(raw_in, model, language,stop_w_flag = True):
    """"receives a raw list of sentences and transforms it into a formatted, tokenized and word2vec friendly list of sentences
    stop_w_flag = optional argument to disable stop_words
    """""
    text_en = []
    stopWords = set(stopwords.words(language))
    for sentence in raw_in:
        aux = []
        sentence = sentence.lower().split(' ')
        for word in sentence:
            if word.isalpha() == False:
                word = re.sub(r'[^a-z]', '', word)                  #regular expression that removes each non lowercase letter
            if word in model:
                if stop_w_flag == False:
                    aux.append(word)
                elif word not in stopWords:
                    aux.append(word)
        text_en.append(aux)
    return text_en




#-------------------------------TEST ONLY-----------------------------------------------------------------
def embedder(sentence, model, language):
    first_word = 0
    word_emb = []
    sentence_test = stop_words(sentence, language)
    if len(sentence_test) > 1:  # do not apply stopwords for really short sentences
        sentence = sentence_test

    for word in sentence[:len(sentence) - 2]:  # delete \n -> last 2 chars
        if first_word == 0:
            if word in model:
                word_emb = model[word]
                first_word = 1
        else:
            if word in model:
                word_emb = word_emb + model[word]

    return word_emb  # np.append(word_emb,len(sentence))  returns the word embedding plus its size(300 + 1)


def sentence_embedder(sentence_list, model, language):
    emb = []
    for sentence in sentence_list:
        if (len(sentence) > 1):
            emb.append(embedder(sentence, model, language))
    return emb

#----------------------------------------------------------------------------------------------------------