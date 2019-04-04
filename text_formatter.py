from nltk.corpus import stopwords


# Load text and treat
def load_care(input, model, language):
    text_en = []
    stopWords = set(stopwords.words(language))
    for item in input:
        item = item.lower()
        stringlist = []
        aux = []
        item = item.replace("'", '')
        for letter in item:
            if letter == ',' or letter == '.' or letter == ';' or letter == '(' or letter == ')' or letter == '/' or letter == '"""' or letter == '"\"' or letter == '"' or letter == '?' or letter == '!' or letter == '&' or letter == '#' or letter == '[' or letter == ']' or letter == '%' or letter == '-' or letter == ':' or letter == '' or letter == '>' or letter == '<':
                letter = ''
            elif letter.isdigit() == True:
                letter = ''
            stringlist.append(letter)                        # list of chars
        stringlist = ''.join(stringlist)                     # transform into a string

        for word in stringlist.split(' '):                   # remove out of dic and stopwords
            if word in model:
                if word not in stopWords:
                    aux.append(word)

        text_en.append(aux)

    return text_en


def stop_words(sentence,language):
    stopWords = set(stopwords.words(language))
    wordsFiltered = []
    words = sentence.split(' ')
    for w in words:
        # remove manually words with single quote
        w = w.rstrip()
        if w == '  ' or w == ' ' or w == '':
            continue
        elif w not in stopWords:
            wordsFiltered.append(w)
    return wordsFiltered

