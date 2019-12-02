import pandas as pd
import re

from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize


def preprocess_data():
    data = pd.read_csv('data/training_data.csv', encoding="latin1")

    # print(data.head(10))

    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values

    processed_texts = []
    for sentence in range(0, len(x)):
        processed_text = str(x[sentence])

        # Replace @name by USER
        processed_text = re.sub('@[\w\_]+', "USER", processed_text)

        # Replace #name by HASHTAG
        processed_text = re.sub('#[\w\_]+', "HASHTAG", processed_text)

        # Replace https by URL
        processed_text = re.sub('https[\w\./]*', "URL", processed_text)

        # Replace all the special characters by ' '
        processed_text = re.sub(r'\W', ' ', processed_text)

        # Substituting multiple spaces with single space
        processed_text = re.sub(r'\s+', ' ', processed_text, flags=re.I)

        # Remove single space from the start
        processed_text = re.sub('^[\s]+', '', processed_text)

        # Remove all single characters
        processed_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_text)

        # Remove single characters from the start
        processed_text = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_text)

        # Converting to Lowercase
        processed_text = processed_text.lower()

        # Remove more than 2 repetition of letters in word
        pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
        processed_text = pattern.sub(r"\1", processed_text)

        processed_texts.append(processed_text)

        # print(processed_text)
        #
        # print("\n")

    return processed_texts, y


def read_word_label_file():
    f = open('data/train_conll.txt', encoding="latin1")
    data = f.read()

    sentences = []
    sentence = []

    lines = data.split('\n')

    for line in lines:
        if line == '':
            sentences.append(sentence)
            sentence = []

        else:
            words = line.split('\t')

            if len(words) == 2 and words[0] != 'meta':
                sentence.append(tuple(words))

    return sentences


def get_tokenized_sentence_list(sentence_list):
    tokenized_sentence_list = []

    for sentence in sentence_list:
        tokens = word_tokenize(sentence)
        tokenized_sentence_list.append(tokens)

    return tokenized_sentence_list


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


def get_eng_words(sentence):
    eng_words = []

    for word_senti_tuple in sentence:
        if word_senti_tuple[1] == "Eng":
            eng_words.append(word_senti_tuple[0])

    return eng_words


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
        return []


def get_english_senti_scores(sentences):
    ps = PorterStemmer()

    senti_scores = []
    for sentence in sentences:
        words = get_eng_words(sentence)
        words = [ps.stem(x) for x in words]

        pos_val = pos_tag(words)

        senti_vals = [get_sentiment(x, y) for (x, y) in pos_val]
        print(words)
        print(senti_vals)

        sum_sentivals = calculate_sentival_sum(senti_vals)
        print(sum_sentivals)

        senti_scores.append(sum_sentivals)

    return senti_scores


def get_senti_english_scores(tokenized_sentences):
    ps = PorterStemmer()

    senti_scores = []
    for sentence_tokens in tokenized_sentences:
        sentence_tokens = [ps.stem(x) for x in sentence_tokens]

        pos_val = pos_tag(sentence_tokens)

        senti_vals = [get_sentiment(x, y) for (x, y) in pos_val]
        print(sentence_tokens)
        print(senti_vals)

        sum_sentivals = calculate_sentival_sum(senti_vals)
        print(sum_sentivals)

        senti_scores.append(sum_sentivals)

    return senti_scores


def get_hindi_senti_scores(sentence_list):
    hindi_score_data = pd.read_csv('Hinglish_Profanity_List.csv', encoding="latin1")

    senti_scores = []
    for sentence in sentence_list:

        sum_scores = 0
        for hindi_word, score in zip(hindi_score_data['hindi'], hindi_score_data['profanity']):

            r = re.compile(r'\b%s\b' % hindi_word, re.I)
            if r.search(sentence.lower()) is not None:

                sum_scores += int(score)

        senti_scores.append(sum_scores)

    return senti_scores


def main():
    processed_texts, sentiments = preprocess_data()

    # sentences = read_word_label_file()
    # english_senti_scores = get_english_senti_scores(sentences)

    # tokenized_sentence_list = get_tokenized_sentence_list(processed_texts)

    # get_senti_english_scores(tokenized_sentence_list)

    hindi_scores = get_hindi_senti_scores(processed_texts)

    for i, score in enumerate(hindi_scores):
        if score > 0:
            print(processed_texts[i] + " ---- " + sentiments[i] + " ----- " + str(score))


if __name__ == "__main__":
    main()
