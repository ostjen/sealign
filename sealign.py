from embedding_related import *
from text_formatter import *
import sys


src_path = '~/desktop/embeddings/wiki.en.align.vec'
tgt_path = '~/desktop/embeddings/wiki.de.align.vec'
nmax = 150000
print('\ncarregando modelos ...')
src_model = gensim.models.KeyedVectors.load_word2vec_format(src_path, limit=nmax)
tgt_model = gensim.models.KeyedVectors.load_word2vec_format(tgt_path, limit=nmax)

print('\nsource language')
src_l = input("Input ")
print('\ntarget language')
tgt_l = input("Input ")

src_sentences = load_care(sys.argv[1],src_l)
tgt_sentences = load_care(sys.argv[2],tgt_l)

generate_matrix (src_model, tgt_model, src_sentences, tgt_sentences)







