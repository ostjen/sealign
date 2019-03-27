from embedding_related import *
from text_formatter import *
import numpy as np
import sys

src_path = '~/desktop/embeddings/wiki.en.align.vec'
tgt_path = '~/desktop/embeddings/wiki.es.align.vec'

en_in = open('./samples/test-enes.en', encoding='utf8').readlines()
es_in = open('./samples/test-enes.es', encoding='utf8').readlines()

#raw input will be used later
english_text = load_care(en_in,'english')
spanish_text = load_care(es_in,'spanish')

print('\nloading models ...')
src_model = gensim.models.KeyedVectors.load_word2vec_format(src_path)
print('\nsrc model loaded')
tgt_model = gensim.models.KeyedVectors.load_word2vec_format(tgt_path)
print('\ndone')

matrix = generate_matrix(src_model, tgt_model, english_text[:50], spanish_text[:50],csv = 'yes')
spanish_index = np.argmin(matrix,axis = 0)
out = open('aligned_out2.txt',"w+",encoding = 'utf-8')
for ind in range(0,len(en_in[:50])):
    aux = en_in[ind].replace('\n','\t')
    print(aux + es_in[spanish_index[ind]])
    out.write(aux)
    out.write(es_in[spanish_index[ind]])








