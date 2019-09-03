import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer
text_of_words = []
stop_words = set(stopwords.words('english'))

def get_data():

    file_path = "./testdata.txt"#str(input("Enter file path: "))
    f = open(file_path, 'r')
    text_data = f.read()
    text_data = text_data.lower()
    td = text_data.strip('\n')
    #word_list = td.split(' ')
    clean_value = clean_text(td)
    get_top_words([clean_value])


def clean_text(words):

    word_list = []
    word_list = re.sub(r'\n|[0-9]', '',words)
    word_list = re.sub('[^a-zA-z]', ' ', word_list)
    word_list = word_list.split(' ')
    word_list = [word for word in word_list if word is not '']
    lem = WordNetLemmatizer()
    text_of_words = [word for word in word_list if word not in stop_words ]#[lem.lemmatize(word, pos = 'v') for word in word_list if word not in stop_words]
    clean_words = ' '
    clean_words = clean_words.join(text_of_words)
    return clean_words
    #createWordCloud(text_of_words)
    #print(text_of_words)

def createWordCloud(words):
    wordcloud = WordCloud(background_color = 'black', stopwords = stop_words, max_words = 100, max_font_size = 75).generate(str(words))
    print(wordcloud)
    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

def get_top_words(text):


    vectorize1 = CountVectorizer(ngram_range = (1, 1)).fit(text)#tokenizes and builds vocabulary
    vector1 = vectorize1.transform(text)#calulates (encodes the document)
    sum_vector1 = vector1.sum(axis = 0)#display the count of the words in an array format!
    words_freqdict1 = [(word, sum_vector1[0, idx]) for word, idx in vectorize1.vocabulary_.items()]#Count of every "n gram(word) combination" is paired
    words_freq1 = sorted(words_freqdict1, key = lambda x : x[1], reverse = True)#sort according to values of the key in reverse order
    print(words_freq1[ : 5])

    vectorize2 = CountVectorizer(ngram_range = (2, 2)).fit(text)
    vector2 = vectorize2.transform(text)
    sum_vector2 = vector2.sum(axis = 0)
    word_freqdict2 = [(word, sum_vector2[0, idx]) for word, idx in vectorize2.vocabulary_.items()]
    words_freq2 = sorted(word_freqdict2, key = lambda x : x[1], reverse = True)
    print(words_freq2[ : 5])

    vectorize3 = CountVectorizer(ngram_range = (3, 3)).fit(text)
    vector3 = vectorize3.transform(text)
    sum_vector3 = vector3.sum(axis = 0)
    word_freqdict3 = [(word, sum_vector3[0, idx]) for word, idx in vectorize3.vocabulary_.items()]
    word_freq3 = sorted(word_freqdict3, key = lambda x : x[1], reverse = True)
    print(word_freq3[ : 5])
    #vector_v = vector.toarray()
    #print("vocab: \n", vectorize.vocabulary_)
    #print("v: \n", vector)
    #print(vector_v)

get_data()
