import pandas as pd
import re

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from nltk.corpus import words
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from sklearn.metrics import confusion_matrix
import time


# preprocess data
def preprocess_data(data):
    for x in range(0, len(data)):

        data["Sentence"][x] = str(data["Sentence"][x])

        # Replace @name by USER
        data["Sentence"][x] = re.sub('@[\w\_]+', "USER", data["Sentence"][x])

        # Replace #name by HASHTAG
        data["Sentence"][x] = re.sub('#[\w\_]+', "HASHTAG", data["Sentence"][x])

        # Replace https by URL
        data["Sentence"][x] = re.sub('https[\w\./]*', "URL", data["Sentence"][x])

        # Replace all the special characters by ' '
        data["Sentence"][x] = re.sub(r'\W', ' ', data["Sentence"][x])

        # Substituting multiple spaces with single space
        data["Sentence"][x] = re.sub(r'\s+', ' ', data["Sentence"][x], flags=re.I)

        # Remove single space from the start
        data["Sentence"][x] = re.sub('^[\s]+', '', data["Sentence"][x])

        # Remove all single characters
        data["Sentence"][x] = re.sub(r'\s+[a-zA-Z]\s+', ' ', data["Sentence"][x])

        # Remove single characters from the start
        data["Sentence"][x] = re.sub(r'\^[a-zA-Z]\s+', ' ', data["Sentence"][x])

        # Converting to Lowercase
        data["Sentence"][x] = data["Sentence"][x].lower()

        # Remove more than 2 repetition of letters in word
        pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
        data["Sentence"][x] = pattern.sub(r"\1", data["Sentence"][x])

        if (data["Sentiment"][x] == "positive"):

            data["Sentiment"][x] = "1"

        elif (data["Sentiment"][x] == "negative"):

            data["Sentiment"][x] = "2"

        elif (data["Sentiment"][x] == "neutral"):

            data["Sentiment"][x] = "0"

    return data


# get hindi token list and english token list from the sentences
def get_tokenized_sentence_list(sentence_list):
    sentence_list_with_english_tokens = []
    sentence_list_with_hindi_tokens = []

    nltk_words_set = set(words.words())

    for sentence in sentence_list:
        tokens = sentence.split()

        english_tokens = []
        hindi_tokens = []
        for token in tokens:
            if token in nltk_words_set:
                english_tokens.append(token)
            else:
                hindi_tokens.append(token)

        sentence_list_with_english_tokens.append(english_tokens)
        sentence_list_with_hindi_tokens.append(hindi_tokens)

    return sentence_list_with_english_tokens, sentence_list_with_hindi_tokens


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def get_sentiment(word, tag):
    """ returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. """

    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []

    lemmatizer = WordNetLemmatizer()

    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score()]


# calculate sum of senti scores for each words in a sentence
def calculate_sentival_sum(senti_vals):
    # append only non empty lists
    filtered_senti_vals = []
    for vals in senti_vals:
        if vals:
            filtered_senti_vals.append(vals)

    # sum all the list values
    if filtered_senti_vals:
        return [sum(x) for x in zip(*filtered_senti_vals)]

    else:
        return [0, 0, 0]


# normalize senti scores
def normalize_values(value_list):
    sum_val = 0
    for value in value_list:
        sum_val += value

    if sum_val != 0:
        for i in range(0, len(value_list)):
            value_list[i] = value_list[i] / sum_val

    return value_list


# get combined senti scores from hindi and english senti scores
def get_combined_normalized_list_of_values(list1, list2):
    normalized_values_list = []
    for values1, values2 in zip(list1, list2):
        sum_senti_values = [x + y for x, y in zip(values1, values2)]
        normalized_values_list.append(normalize_values(sum_senti_values))

    return normalized_values_list

# get english senti scores for all sentences
def get_english_senti_scores(tokenized_sentences):
    ps = PorterStemmer()

    senti_scores = []
    for sentence_tokens in tokenized_sentences:
        sentence_tokens = [ps.stem(x) for x in sentence_tokens]

        pos_val = pos_tag(sentence_tokens)

        senti_vals = [get_sentiment(x, y) for (x, y) in pos_val]
        # print(sentence_tokens)
        # print(senti_vals)

        sum_sentivals = calculate_sentival_sum(senti_vals)

        senti_scores.append(sum_sentivals)

    return senti_scores


# get hindi profanity scores of all sentences
def get_hindi_profanity_scores(tokenized_sentences):
    hindi_score_data = pd.read_csv('data/Hinglish_Profanity_List.csv', encoding="latin1")

    profanity_scores = []
    for sentence_tokens in tokenized_sentences:

        sum_scores = 0
        for token in sentence_tokens:

            for hindi_word, score in zip(hindi_score_data['Hindi'], hindi_score_data['Profanity']):

                if token == hindi_word:
                    sum_scores += int(score)
                    break

        profanity_scores.append(sum_scores)

    return profanity_scores


# load and parse hindi senti wordnet
def parse_hindi_senti_wordnet(filename):
    f = open(filename, encoding="utf-8")

    data = f.read()

    pos_word_tuple_list = []

    lines = data.split('\n')

    for line in lines:

        line = line.strip()
        if line != '' and line[0] != '#':

            pos_word = line.split('\t')

            if len(pos_word) == 2:
                pos_word_tuple_list.append(tuple(pos_word))

    return pos_word_tuple_list


# calculate hindi senti scores of all sentences
def get_hindi_senti_scores(tokenized_sentences):
    senti_scores_list = []

    hindi_senti_wordnet_dict = {'negative': parse_hindi_senti_wordnet('data/Hindi_SentiWordNet/HN_NEG.txt'),
                                'positive': parse_hindi_senti_wordnet('data/Hindi_SentiWordNet/HN_POS.txt'),
                                'neutral': parse_hindi_senti_wordnet('data/Hindi_SentiWordNet/HN_NEU.txt')}

    for sentence_tokens in tokenized_sentences:
        pos_score = 0
        neg_score = 0
        neu_score = 0

        for token in sentence_tokens:

            converted_hindi_word = transliterate(token, sanscript.ITRANS, sanscript.DEVANAGARI)
            found = False

            for pos_word_tuple in hindi_senti_wordnet_dict['positive']:

                if converted_hindi_word == pos_word_tuple[1]:
                    pos_score += 1
                    found = True
                    break

            if not found:
                for pos_word_tuple in hindi_senti_wordnet_dict['negative']:

                    if converted_hindi_word == pos_word_tuple[1]:
                        neg_score += 1
                        found = True
                        break

            if not found:
                for pos_word_tuple in hindi_senti_wordnet_dict['neutral']:

                    if converted_hindi_word == pos_word_tuple[1]:
                        neu_score += 1
                        found = True
                        break

        senti_scores_list.append([pos_score, neg_score, neu_score])

    return senti_scores_list

# train dnn classifier
def train(train_sentences, train_sentiments,  step_size, total_steps):

    train_sentences_with_english_tokens, train_sentences_with_hindi_tokens = get_tokenized_sentence_list(
        train_sentences)

    eng_senti_train_scores = get_english_senti_scores(train_sentences_with_english_tokens)
    hindi_senti_train_scores = get_hindi_senti_scores(train_sentences_with_hindi_tokens)

    combined_senti_train_scores = get_combined_normalized_list_of_values(eng_senti_train_scores,
                                                                         hindi_senti_train_scores)

    hindi_profanity_train_score_list = get_hindi_profanity_scores(train_sentences_with_hindi_tokens)
    hindi_profanity_train_scores = np.array(hindi_profanity_train_score_list)

    positive_senti_train_score = []
    negative_senti_train_score = []
    objective_senti_train_score = []

    for x in combined_senti_train_scores:
        positive_senti_train_score.append(x[0])
        negative_senti_train_score.append(x[1])
        objective_senti_train_score.append(x[2])

    positive_senti_train_scores = np.array(positive_senti_train_score)
    negative_senti_train_scores = np.array(negative_senti_train_score)
    objective_senti_train_scores = np.array(objective_senti_train_score)

    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        {'sentence': train_sentences, 'senti_pos_score': positive_senti_train_scores,
         'senti_neg_score': negative_senti_train_scores, 'senti_obj_score': objective_senti_train_scores,
         'hin_prof_score': hindi_profanity_train_scores}, train_sentiments,
        batch_size=512, num_epochs=None, shuffle=True)

    predict_train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        {'sentence': train_sentences, 'senti_pos_score': positive_senti_train_scores,
         'senti_neg_score': negative_senti_train_scores, 'senti_obj_score': objective_senti_train_scores,
         'hin_prof_score': hindi_profanity_train_scores}, train_sentiments, shuffle=False)

    embedding_feature = hub.text_embedding_column(
        key='sentence',
        module_spec="https://tfhub.dev/google/universal-sentence-encoder/2",
        trainable=True)

    senti_pos_score = tf.feature_column.numeric_column(key='senti_pos_score')
    senti_neg_score = tf.feature_column.numeric_column(key='senti_neg_score')
    senti_obj_score = tf.feature_column.numeric_column(key='senti_obj_score')
    hin_prof_score = tf.feature_column.numeric_column(key='hin_prof_score')

    dnn = tf.compat.v1.estimator.DNNClassifier(
        hidden_units=[512, 128],
        feature_columns=[embedding_feature, senti_pos_score, senti_neg_score, senti_obj_score, hin_prof_score],
        n_classes=3,
        label_vocabulary=["0", "1", "2"],
        activation_fn=tf.nn.relu,
        dropout=0.1,
        optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate=0.005))

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    for step in range(0, total_steps + 1, step_size):
        print()
        print('-' * step_size)
        print('Training for step =', step)
        start_time = time.time()
        dnn.train(input_fn=train_input_fn, steps=step_size)
        elapsed_time = time.time() - start_time
        print('Train Time (s):', elapsed_time)
        print('Eval Metrics (Train):', dnn.evaluate(input_fn=predict_train_input_fn))

    return dnn


# test using dnn classifier
def test(test_sentences, test_sentiments, dnn):

    test_sentences_with_english_tokens, test_sentences_with_hindi_tokens = get_tokenized_sentence_list(test_sentences)

    eng_senti_test_scores = get_english_senti_scores(test_sentences_with_english_tokens)
    hindi_senti_test_scores = get_hindi_senti_scores(test_sentences_with_hindi_tokens)

    combined_senti_test_scores = get_combined_normalized_list_of_values(eng_senti_test_scores, hindi_senti_test_scores)

    hindi_profanity_test_score_list = get_hindi_profanity_scores(test_sentences_with_hindi_tokens)
    hindi_profanity_test_scores = np.array(hindi_profanity_test_score_list)

    positive_senti_test_score = []
    negative_senti_test_score = []
    objective_senti_test_score = []

    for x in combined_senti_test_scores:
        positive_senti_test_score.append(x[0])
        negative_senti_test_score.append(x[1])
        objective_senti_test_score.append(x[2])

    positive_senti_test_scores = np.array(positive_senti_test_score)
    negative_senti_test_scores = np.array(negative_senti_test_score)
    objective_senti_test_scores = np.array(objective_senti_test_score)

    predict_test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        {'sentence': test_sentences, 'senti_pos_score': positive_senti_test_scores,
         'senti_neg_score': negative_senti_test_scores, 'senti_obj_score': objective_senti_test_scores,
         'hin_prof_score': hindi_profanity_test_scores}, test_sentiments, shuffle=False)

    print('Eval Metrics (Test):', dnn.evaluate(input_fn=predict_test_input_fn))

    predictions = dnn.predict(input_fn=predict_test_input_fn)

    predicted_sentiments = []
    for prediction in list(predictions):
        predicted_sentiments.append(str(prediction['classes'][0].decode('ascii')))

    cm = confusion_matrix(test_sentiments, predicted_sentiments, ['1', '2', '0'])
    print("Confusion Matrix:\n %s \n %s" % (['positive', 'negative', 'neutral'], cm))


def main():
    # training
    train_data = pd.read_csv('data/training_data.csv', encoding="latin1")
    train_data = preprocess_data(train_data)

    train_sentences = train_data['Sentence'].values
    train_sentiments = train_data['Sentiment'].values

    dnn = train(train_sentences, train_sentiments, 30, 120)

    # testing
    test_data = pd.read_csv('data/test_data.csv', encoding="latin1")
    test_data = preprocess_data(test_data)

    test_sentences = test_data['Sentence'].values
    test_sentiments = test_data['Sentiment'].values

    test(test_sentences, test_sentiments, dnn)


if __name__ == "__main__":
    main()
