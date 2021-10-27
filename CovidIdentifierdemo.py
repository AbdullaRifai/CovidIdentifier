import csv
import sys

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

import string
import re
import random

import gensim
from gensim import corpora, models, similarities
from gensim.models import Word2Vec

# for demo
import tkinter
from functools import partial

# Input: sentence
# Filters stopwords in a sentence and returns array of the Lemmatize of the words


def filter_sentence(text):
    # removes punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    lem = WordNetLemmatizer()
    tokenized_sent = word_tokenize(text, "english")

    filtered_sent = []
    stop_words = set(stopwords.words("english"))
    # add stop_words as they give problems
    stop_words.add('go')
    for w in tokenized_sent:
        word = lem.lemmatize(w.lower(), "v")
        if word not in stop_words:
            filtered_sent.append(word)
    # for i, word in enumerate(filtered_sent):
    #     filtered_sent[i] = lem.lemmatize(word.lower(), "v")

    return filtered_sent


# Function to get three largest elements
def get3largest(arr, arr_size):
    # There should be atleast three
    # elements
    if arr_size < 3:
        print(" Invalid Input ")
        return

    third = first = second = -sys.maxsize

    for i in range(0, arr_size):

        if arr[i] > first:

            third = second
            second = first
            first = arr[i]

        elif arr[i] > second:

            third = second
            second = arr[i]

        elif arr[i] > third:
            third = arr[i]

    return first, second, third


# Input: array of words (representing a sentence)
# to filter nouns
def filter_nouns(words):
    def is_noun(pos): return pos[:2] == 'NN'
    nouns = [word for (word, pos) in nltk.pos_tag(words) if is_noun(pos)]
    return nouns


# processes one of the csv files we have like fixed_processed_kaggle.csv
def process_csv(name):
    covidVecs = []
    with open(name, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        tweet_id = []
        for row in csv_reader:
            if row[3] not in tweet_id:
                # process everything in lower case to remove bias
                sent = re.sub(r"http\S+", " ", row[1].lower())
                # regex = re.compile('[^a-zA-Z0-9.,;\s]')
                # sent = regex.sub('', sent)
                covidVecs.append(word_tokenize(sent, "english"))
                tweet_id.append(row[3])

    return covidVecs


def produce_word2vec_model(words, filename="Covid_Identifying_Model"):
    model = gensim.models.Word2Vec(words, min_count=1, size=32)
    model.save(filename)
    return model


def covid_metrics(text):
    model = Word2Vec.load("Patient-Doctor-casual_model")
    model.init_sims(replace=True)
    # Not using get_nouns because although covid is a noun the nltk model isn't recognizing it (new term)
    words = filter_sentence(text)
    metrics = []
    for word in words:
        # all the words are lower case to improve accurasey
        # print(word)
        try:
            best_metric = [model.wv.similarity(word, "corona"),
                           model.wv.similarity(word, "covid19"),
                           model.wv.similarity(word, "covid-19"),
                           model.wv.similarity(word, "covid"),
                           model.wv.similarity(word, "coronavirus")]
            # print(max(best_metric))
            metrics.append(max(best_metric))
        except Exception as e:
            metrics.append(0)
    # meaning all the words are useless and have been removed by filter_sentence (thus casual)
    if not metrics:
        return 0
    # If there is a key word from the above list we know its for the covid agent
    if max(metrics) > 0.95 or text.find("face mask") >= 0:
        return 1
    else:
        confidence = sum(metrics) / len(metrics)
        return confidence


def display():
    top = tkinter.Tk()
    top.configure(bg='gray33')
    top.title("metrics")
    # Increasing the font
    fontStyle = tkinter.font.Font(family="Lucida Grande", size=18)

    def cb(text1, label_list, event=None):  # event is passed by binders.
        index = label_list.size()
        input_text = str(text1.get())
        metric_covid = better_metrics(input_text)
        text1 = "word2vec model1: "
        if metric_covid > 0.5:
            text1 = text1 + "covid agent"
        else:
            text1 = text1 + "casual agent"
        if index > 8:
            for i in range(4):
                label_list.delete(0)
        label_list.insert(index, "Human:    " + input_text)
        label_list.insert(index + 1, text1)
        label_list.insert(index + 2, "metric: " + str(metric_covid))
        label_list.insert(index + 3, "")
        return

    messages_frame = tkinter.Frame(top).pack()
    my_msg = tkinter.StringVar()  # For the messages to be sent.
    # To navigate through past messages.
    scrollbar = tkinter.Scrollbar(messages_frame)

    # Following will contain the messages.
    msg_list = tkinter.Listbox(
        messages_frame, height=25, width=130, yscrollcommand=scrollbar.set, font=fontStyle)
    scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)
    msg_list.pack(side=tkinter.BOTTOM, fill=tkinter.BOTH)
    msg_list.pack()
    msg_list.configure(justify=tkinter.CENTER, bg='gray50')
    msg_list.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

    entry_field = tkinter.Entry(top, width=60, textvariable=my_msg)

    # Make the parameters built in and we just pass call_result to the event and button
    call_result = partial(cb, my_msg, msg_list)
    tkinter.Button(top, text="Enter", command=call_result).pack()

    entry_field.bind("<Return>", call_result)
    entry_field.pack()

    tkinter.mainloop()  # Starts GUI execution.


# second metric method (works better with longer sentences as apposed to the the other one.
# I plan on making a voting system once I come up with a third algorithm
def better_metrics(text):
    model = Word2Vec.load("Patient-Doctor-casual_model")
    model.init_sims(replace=True)
    # Not using get_nouns because although covid is a noun the nltk model isn't recognizing it (new term)
    words = filter_sentence(text)
    metrics = []
    for word in words:
        # all the words are lower case to improve accurasey
        # print(word)
        try:
            best_metric = [model.wv.similarity(word, "corona"),
                           model.wv.similarity(word, "covid"),
                           model.wv.similarity(word, "covid19"),
                           model.wv.similarity(word, "covid-19"),
                           model.wv.similarity(word, "coronavirus")]
            # print(max(best_metric))
            metrics.append(max(best_metric))
        except Exception as e:
            print(e)
            metrics.append(0)

    if len(metrics) > 3:
        largest, second, third = get3largest(metrics, len(metrics))
        if max(metrics) > 0.95 or text.find("face mask") >= 0:
            return 1
        elif largest > 0.75 and second > 0.6:
            return largest
        else:
            return third
    else:
        return covid_metrics(text)


if __name__ == "__main__":
    model = Word2Vec.load("Patient-Doctor-casual_model")
    display()
