import pandas as pd

from code_mixed_semeval import *


def main():
    data = pd.read_csv('data/IIITH_Codemixed.csv', encoding="latin1")
    data = preprocess_data(data)
    data = data.sample(frac=1)

    sentences = data['Sentence'].values
    sentiments = data['Sentiment'].values

    # training
    train_sentences = sentences[:3103]
    train_sentiments = sentiments[:3103]

    dnn = train(train_sentences, train_sentiments, 6, 36)

    # testing
    test_sentences = sentences[3103:]
    test_sentiments = sentiments[3103:]

    test(test_sentences, test_sentiments, dnn)


if __name__ == "__main__":
    main()
