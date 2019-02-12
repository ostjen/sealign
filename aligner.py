import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


def split_sentence(input_file , output_name):
	
	file = open(input_file, "r").read()
	out_list = sent_tokenize(file)
	out_file = open(output_name,"a+")
	for element in out_list:
		out_file.write(element + '￭')
	
	out_file.close()



split_sentence("/Users/leonardo/desktop/embedding_align/samples/text1_en.txt","teste.txt")
