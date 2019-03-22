from nltk.corpus import stopwords

def load_care(path,language):
    text_en = []
    aux = open(path, encoding='utf8').readlines()
    for item in aux:
        item = item.lower()
        stringlist = []
        item = item.replace("'", '')
        for letter in item:
            if letter == ',' or letter == '.' or letter == ';' or letter == '(' or letter == ')' or letter == '/' or letter == '"""' or letter == '"\"' or letter == '"' or letter == '?' or letter == '!' or letter == '&' or letter == '#' or letter == '[' or letter == ']' or letter == '%' or letter == '-' or letter == ':' or letter == '' or letter == '>' or letter == '<':
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
    sentence = sentence.split(' ')
    for w in sentence:
        w = w.rstrip()
        if w == '   ' or w == '  ' or w == ' ' or w == '':
            continue
        elif w not in stopWords:
            wordsFiltered.append(w)

    if(len(wordsFiltered) >= 1):
        return ' '.join(wordsFiltered)
    else:
        return sentence

