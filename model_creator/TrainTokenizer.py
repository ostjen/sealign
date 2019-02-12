#punkt trainer - by Chaitra

selected_language = ''
abbreviations_file = ''

def loadNltkdependdencies():
 print('')
 print('loading dependencies...') 
 print('-----------------------------------------------------------------------')
 nltk.download('punkt')
 print('-----------------------------------------------------------------------')
 print('loading dependencies completed')
 print('')
 print('') 

def loadLanguageList():
 #variables
 lang_count = 0 
 input_language = ''
 selected_language_ = ''
 #language list
 languages_ = ['czech','danish','dutch','english','estonian','finnish','french','german',
              'greek','italian','norwegian','polish','portuguese','slovene','spanish','swedish','turkish']  
 print('Language-List')
 print('-----------------------------------------------------------------------')
 
 for language_ in languages_:
  lang_count += 1
  print(str(lang_count) + '.' + language_)
 print('-----------------------------------------------------------------------')
 input_language = input('Select A Language:')

 try:
  if int(input_language) > len(languages_):
   pass
  else:
   selected_language_ = languages_[int(input_language)-1]
 except ValueError:
  pass
 return selected_language_

def DisplayProgramHeader():
 print('Punkt Tokenizer Trainer - v1.1 - By Chaitra Dangat')
 print('')

def train_tokenizer(trainfile,abbreviationfile,modelfile):
 k = 0
 skipped_ = 0
 custom_ = 0
 
 punkt = PunktTrainer()
 input_ = codecs.open(trainfile, encoding='utf-8')
 
 for sentence in input_:
  k+=1
  if k%100 == 0:
   print('trained from sentences :' + str(k))
  try:
   punkt.train(sentence, finalize=False, verbose=False)
  except:
   skipped_ += 1
 
 input_.close()
 
 if abbreviationfile !='':
  abbreviations_ = codecs.open(abbreviationfile,encoding='utf-8') 
  for abbr in abbreviations_:
   try:
    punkt.train('Start ' + abbr + '. End.' ,finalize=False, verbose=False)
    custom_ += 1
   except:
    pass
  abbreviations_.close()
  
 punkt.finalize_training(verbose=False)
 
 model = PunktSentenceTokenizer(punkt.get_params())
 model_output = codecs.open(modelfile,mode='wb')
 pickle.dump(model,model_output,protocol=pickle.HIGHEST_PROTOCOL)
 model_output.close()
 
 print('')
 print(str(skipped_) + ' sentences skipped')
 print(str(custom_) + ' custom abbreviations added')
 
 
 
import re
import os
import sys 
import codecs
import nltk
loadNltkdependdencies()
import nltk.data
from nltk.tokenize.punkt import PunktTrainer,PunktSentenceTokenizer,PunktParameters
import pickle

DisplayProgramHeader()

selected_language = loadLanguageList()
if selected_language != '':
 print('language selected: '+selected_language)
 print('')
else:
 print('invalid selection!')
 quit() 

#corpus file path
corpus_file = input('Enter the '+ selected_language +' text file path :')

#abbreviations file path
abbreviations_file = input('Enter the ' + selected_language + ' abbreviations file path(optional):')
print('Abbreviations file selected:' + abbreviations_file)
print('')

#output folder path
output_folder = input('Enter the output folder path :')
model_file = output_folder + '\\' + selected_language + '.pickle'

print('Saving the model...')
train_tokenizer(corpus_file,abbreviations_file,model_file)
print('Saving the model Completed')

print('')
print('training completed..')