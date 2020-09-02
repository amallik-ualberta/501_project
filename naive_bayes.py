import math
import sys

from sklearn.metrics import confusion_matrix

from nltk.corpus import stopwords

# nltk stopwords
nltk_stopwords = set(stopwords.words('english'))


# Class to store Information about each Category from the training data
class CategoryInfo:
    def __init__(self, name, log_prior, word_prob_dictionary):
        self.name = name                                    # Category name
        self.log_prior = log_prior                          # Category log prior probability
        self.word_prob_dictionary = word_prob_dictionary    # Dictionary containing log probability of each word of the Category


# Class to store the trained Model
class Model:
    def __init__(self, category_info_list, vocabulary):
        self.category_info_list = category_info_list    # List of CategoryInfo Class objects
        self.vocabulary = vocabulary                    # Vocabulary using all words in training data


# Class to store output data
class OutputData:
    def __init__(self, original_label, assigned_label, text):
        self.original_label = original_label    # true label
        self.assigned_label = assigned_label    # predicted label
        self.text = text                        # test


# write information regarding each sample to output file
def write_output(output_file, output_data):

    # format the output file name
    output_file = "output_" + output_file + ".csv"

    f = open(output_file, "w", encoding="latin1")

    # write the header
    f.write("original label,classifier-assigned label,text" + "\n")

    # write true label, predicted label and text for all samples one by one
    for each in output_data:
        f.write(each.original_label + "," + each.assigned_label + "," + each.text + "\n")

    f.close()


# remove nltk stop words from the word list
def remove_stop_words(word_list):
    filtered_word_list = []

    for word in word_list:

        # append only words that are not nltk stopwords in the filtered word list
        if word not in nltk_stopwords:
            filtered_word_list.append(word)

    return filtered_word_list


# get file name excluding extension
def get_file_name_excluding_extension(file_path):
    # for windows machine only, replace '\' with '/' in file path
    file_path = file_path.replace("\\", '/')

    # if file path doesn't contain '/' then get the file name after splitting using '.'
    if file_path.find('/') == -1:
        filename_without_ext = file_path.rsplit('.', 1)[0]

    # if file path contains '/' then split the file name using '/' first and then using '.'
    else:
        filename = file_path.rsplit('/', 1)[1]
        filename_without_ext = filename.rsplit('.', 1)[0]

    return filename_without_ext


# preprocess the data
def preprocess_data(data):

    # get list of samples
    sample_list = data.split('\n')

    # remove the header
    del sample_list[0]

    processed_sample_list = []

    for sample in sample_list:

        if len(sample) > 0:

            # split sample into (category, text) tuple
            category_text_tuple = tuple(sample.split(',', 1))

            x = category_text_tuple[0]

            y = category_text_tuple[1]

            category_text_tuple = tuple((y,x))

            # append tuple to the sample list
            processed_sample_list.append(category_text_tuple)

    return processed_sample_list


# train a model using the training data
def train(train_set, category_set):
    model = None

    # value of alpha for Alpha additive smoothing
    alpha = 0.1

    # list of words in the training data
    word_list = []

    for x in train_set:
        # # tokenize the text using nltk word_tokenize
        # tokens = word_tokenize(x[1])

        # tokenize by splitting words with whitespace
        tokens = x[1].split()

        # remove stopwords from the token list
        tokens = remove_stop_words(tokens)

        word_list += tokens

    # get the vocabulary by removing duplicate words in the word list
    vocabulary = set(word_list)

    category_info_list = []
    for category in category_set:

        category_count = 0
        word_prob_dictionary = {}

        # contains all the tokens of all the docs in the category by appending tokens of all docs one by one
        bigdoc = []

        for x in train_set:

            if x[0] == category:
                # # tokenize the text using nltk word_tokenize
                # tokens = word_tokenize(x[1])

                # tokenize by splitting words with whitespace
                tokens = x[1].split()

                # remove stopwords from the token list
                tokens = remove_stop_words(tokens)

                # append all the tokens of the doc into the bigdoc
                bigdoc += tokens

                category_count += 1

        # calculate log prior probability of the category
        log_prior = math.log2(category_count / len(train_set))

        # total words in the category
        category_word_count = len(bigdoc)

        for word in vocabulary:

            # number of occurrences of the word in all the docs of the category
            word_count = bigdoc.count(word)

            # calculate log likelihood probability of the word using alpha additive smoothing
            log_likelihood = math.log2((word_count + alpha) / (category_word_count + len(vocabulary) * alpha))

            word_prob_dictionary[word] = log_likelihood

        # store Category information and probabilities in the CategoryInfo class
        category_info = CategoryInfo(category, log_prior, word_prob_dictionary)
        category_info_list.append(category_info)

        # store the model in the Model Class
        model = Model(category_info_list, vocabulary)

    return model


# classify test doc using the model
def classify(model, test_doc):

    # set max val to minimum negative value
    max_val = - sys.maxsize - 1

    max_category = None

    # calculate log posterior probability for each category
    for x in model.category_info_list:

        # initialize the sum log probability with prior probability of the Category
        sum_c = x.log_prior

        # # tokenize using nltk word_tokenize
        # tokens = word_tokenize(test_doc)

        # tokenize by splitting words with whitespace
        tokens = test_doc.split()

        # remove stopwords from the token list
        tokens = remove_stop_words(tokens)

        for token in tokens:

            # if word in the vocabulary otherwise ignore
            if token in model.vocabulary:

                # add word log probability to the sum
                sum_c += x.word_prob_dictionary[token]

        # calculate the max probability
        if sum_c > max_val:
            max_val = sum_c

            max_category = x.name

    return max_category


# divide data into 3 folds
def fold_data(data):
    first_split_point = int(len(data) / 3)

    second_split_point = 2 * int(len(data) / 3)

    # fold 1
    fold1 = data[:first_split_point]

    # fold 2
    fold2 = data[first_split_point:second_split_point]

    # fold 3
    fold3 = data[second_split_point:]

    fold_list = [fold1, fold2, fold3]

    # list containing data ranges for each fold as [(1, 568), (569, 1136), (1137, 1705)]
    fold_range_list = [(1, first_split_point),
                       (first_split_point + 1, second_split_point),
                       (second_split_point + 1, len(data))]

    return fold_list, fold_range_list


# 3-fold cross validation using training data
def cross_validation(train_data):
    # get 3 folds and row number ranges of data for those folds
    fold_list, fold_range_list = fold_data(train_data)

    total_accuracy = 0

    # train and validate a model 3 times using 3 folds
    for i in range(0, len(fold_list)):

        train_set = []
        dev_set = []

        print("######## Cross Validation %s ########" % (i + 1))
        print("Row Number 1 : Header")

        for index, fold_range in enumerate(fold_range_list):

            # get the test fold (dev set)
            if index == i:
                dev_set = fold_list[index]

                # adding 1 because the first row is Header in the data
                print("Row Number %s to %s : Dev Set" % (fold_range[0] + 1, fold_range[1] + 1))

            # get train folds
            else:
                train_set += fold_list[index]

                # adding 1 because the first row is Header in the data
                print("Row Number %s to %s : Training Set" % (fold_range[0] + 1, fold_range[1] + 1))

        # train a model using train folds
        model = get_trained_model(train_set)

        correct_category_count = 0

        # classify samples in dev set one by one
        for x in dev_set:

            # classify
            best_category = classify(model, x[1])

            # count the instances where predicted and true category are same
            if x[0] == best_category:
                correct_category_count += 1

        # calculate and print training accuracy
        accuracy = correct_category_count / len(dev_set) * 100
        print("Training Accuracy: %s\n" % accuracy)

        total_accuracy += accuracy

    # calculate and print average training accuracy
    print("Average Training Accuracy: %s\n" % (total_accuracy / len(fold_list)))


# train and return a model on training data
def get_trained_model(train_set):
    category_list = []

    # get the list of all categories
    for x in train_set:
        category_list.append(x[0])

    # convert the category list to a set to remove duplicates
    category_set = set(category_list)

    # train a model on the training data
    model = train(train_set, category_set)

    return model


# test the model on test data
def test(model, test_set):
    true_category_list = []
    predicted_category_list = []
    correct_category_count = 0
    output_data_list = []

    # classify test samples one by one
    for x in test_set:

        # classify the sample
        best_category = classify(model, x[1])

        # store output data to OutputData class
        output_data = OutputData(x[0], best_category, x[1])

        # append output data to the output data list
        output_data_list.append(output_data)

        # append to a list of true categories
        true_category_list.append(x[0])

        # append to a list of predicted categories
        predicted_category_list.append(best_category)

        # to count the instances where predicted category and true category are same
        if x[0] == best_category:
            correct_category_count += 1

    # calculate and print accuracy
    accuracy = correct_category_count / len(test_set) * 100
    print("Test Accuracy: %s" % accuracy)

    # get the category set
    category_set = set(true_category_list)

    # generate and print confusion matrix
    # The order of categories in confusion matrix can change because set is unordered.
    cm = confusion_matrix(true_category_list, predicted_category_list, list(category_set))
    print("Confusion Matrix:\n %s \n %s" % (list(category_set), cm))

    return output_data_list


# evaluate model on evaluation data
def evaluate(model, eval_set):
    # contains output data for all the evaluation samples
    output_data_list = []

    # classify samples one by one
    for x in eval_set:
        # classify the sample
        best_category = classify(model, x[1])

        # store output data in OuputData class
        output_data = OutputData('', best_category, x[1])

        # append output data to output data list
        output_data_list.append(output_data)

    return output_data_list


def main():
    # parse command line arguments (expecting: python3 main.py TRAIN_FILE_PATH TEST_FILE_PATH)
    if len(sys.argv) < 3:
        print("Invalid number of arguments. Use following command pattern:")
        print("python3 main.py TRAIN_FILE_PATH TEST_FILE_PATH EVAL_FILE_PATH")
        return
    else:
        train_file_path = sys.argv[1]
        test_file_path = sys.argv[2]

    # read and preprocess training file
    with open(train_file_path, encoding="latin1") as f:
        train_content = f.read()

        # preprocess training data
    processed_train_sample_list = preprocess_data(train_content)

    # cross validate on training data
    cross_validation(processed_train_sample_list)

    # train model on whole training data
    model = get_trained_model(processed_train_sample_list)

    # read and preprocess test file
    with open(test_file_path, encoding = "latin1") as f:
        test_content = f.read()

        # preprocess test data
        processed_test_sample_list = preprocess_data(test_content)

    # test the model on test data
    output_data_list = test(model, processed_test_sample_list)

    # write test output to output file
    write_output(get_file_name_excluding_extension(test_file_path), output_data_list)


if __name__ == "__main__":
    main()
