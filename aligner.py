import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import sys
import fileinput





def split_sentence(input_file , output_name):	
#transforms a text file into a list of tokenized sentences and generates a backup text file	file = open(input_file, "r").read()	
	file = open(input_file, "r").read()
	file = file.lower()
	file = file.replace('\n',' ')

	out_list = sent_tokenize(file)				#uses nltk models
	out_file = open(output_name,"a+")
	
	for element in out_list:
		out_file.write(element + '￭')
	
	out_file.close()
	
	return out_list
#________________________________________________________________________________________________________________________________________________________________

 #def add_space(input_file):
#	text = open(input_file,"r+")
#	    sys.stdout.write(' {l}'.format(l=line))
#	    sys.stdout.write("{} \n".format(line.rstrip()))


#_________________________________________________________________________________________________________________________________________________________________



split_sentence(sys.argv[1] , sys.argv[2])






