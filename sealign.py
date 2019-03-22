from embedding_related import *
from text_formatter import *
import sys


src_path = '~/desktop/embeddings/wiki.en.align.vec'
tgt_path = '~/desktop/embeddings/wiki.es.align.vec'
nmax = 150000

src_sentences = load_care(sys.argv[1],'english')
tgt_sentences = load_care(sys.argv[2],'spanish')

print('\nloading models ...')
src_model = gensim.models.KeyedVectors.load_word2vec_format(src_path)
print('\nsrc model loaded')
tgt_model = gensim.models.KeyedVectors.load_word2vec_format(tgt_path)

print('\ndone')



generate_matrix (src_model, tgt_model, src_sentences[:50], tgt_sentences[:50])







