from nltk.corpus import stopwords

def load_care(path,language):
    text_en = []
    aux = open(path, encoding='utf8').readlines()
    for item in aux:
        item = item.lower()
        stringlist = []
        item = item.replace("'", '')
        for letter in item:
            if letter == ',' or letter == '.' or letter == ';' or letter == '(' or letter == ')' or letter == '/' or letter == '"""'  or letter == '"' or letter == '?' or letter == '!':
                letter = ''
            elif letter.isdigit() == True:
                letter = ''
            stringlist.append(letter)
        text_en.append(''.join(stringlist))

    for i in range(0,len(text_en)):
        text_en[i] = stop_words(text_en[i],language)


    return text_en


def stop_words(sentence,language):
    stopWords = set(stopwords.words(language))
    wordsFiltered = []
    words = sentence.split(' ')
    for w in words:
        #remove manually words with single quote
        if w == "you're" or w == "i'll" or "we're" == w or w == "i'm" or w == "he's" and w == "she's" and w == "they're" :
            continue

        if w not in stopWords:
            wordsFiltered.append(w)

    if(len(wordsFiltered) >= 1):
        return ' '.join(wordsFiltered)
    else:
        return sentence



