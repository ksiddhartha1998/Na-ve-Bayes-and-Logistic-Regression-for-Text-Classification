import sys
import getopt
import random
import math
import re
from os import listdir
from os.path import isfile, join


class SpamFilter:
    # Variables
    learning_rate = 0.0
    lambda_value = 0.0
    runs = 0
    filter_stop_words = False
    initial_weight = 0.1
    weight_matrix = dict()
    # Stop words and total number of tokens
    stop_words = set()
    tokens = set()
    # Data structures for storing spam and ham tokens
    spam_tokens = dict()
    ham_tokens = dict()

    spam_tokens_dict = dict()  # HashMap < String, HashMap < String, Integer >> ();
    ham_tokens_dict = dict()  # HashMap < String, HashMap < String, Integer >> ();
    spam_set = set()  # new HashSet<String>();
    ham_set = set()  # new HashSet<String>();
    file_name_set = set()  # new HashSet<String>();


def read_file(txt_file_path):
    # Open File at path in read only mode
    txt_file = open(txt_file_path, 'r')
    lines = txt_file.read().split("\n")
    txt_file.close()
    return lines


def clean_string(string_data):
    string_data = re.sub("<.*?>", "", string_data)
    string_data = re.sub("'s", "", string_data)
    string_data = re.sub("[^a-zA-Z]", "", string_data)
    return string_data


def read(mail_path):
    mails = [f for f in listdir(mail_path) if isfile(join(mail_path, f))]
    for file in mails:
        file = mail_path + "/" + file
        for line in read_file(file):
            for word in line.lower().strip().split(" "):
                clean_data = clean_string(word)
                if clean_data is not None and clean_data != "":
                    SpamFilter.tokens.add(clean_data)


def find_tokens(mail_path, is_spam):
    mails = [f for f in listdir(mail_path) if isfile(join(mail_path, f))]
    for file in mails:
        SpamFilter.file_name_set.add(file)
        for line in read_file(mail_path + "/" + file):
            for word in line.lower().strip().split(" "):
                clean_data = clean_string(word)
                if clean_data is not None and clean_data != "":
                    input_dict = dict()
                    if is_spam:
                        SpamFilter.spam_set.add(file)
                        if SpamFilter.tokens.__contains__(clean_data):
                            if SpamFilter.spam_tokens.keys().__contains__(clean_data):
                                SpamFilter.spam_tokens[clean_data] = SpamFilter.spam_tokens.get(clean_data) + 1
                            else:
                                SpamFilter.spam_tokens[clean_data] = 1
                            if input_dict.keys().__contains__(clean_data):
                                input_dict[clean_data] = input_dict.get(clean_data) + 1
                            else:
                                input_dict[clean_data] = 1
                            SpamFilter.spam_tokens_dict[file] = input_dict
                    else:
                        SpamFilter.ham_set.add(file)
                        if SpamFilter.tokens.__contains__(clean_data):
                            if SpamFilter.ham_tokens.keys().__contains__(clean_data):
                                SpamFilter.ham_tokens[clean_data] = SpamFilter.ham_tokens.get(clean_data) + 1
                            else:
                                SpamFilter.ham_tokens[clean_data] = 1
                            if input_dict.keys().__contains__(clean_data):
                                input_dict[clean_data] = input_dict.get(clean_data) + 1
                            else:
                                input_dict[clean_data] = 1
                            SpamFilter.ham_tokens_dict[file] = input_dict


def apply_logistic_regression(net):
    if net < -100:
        return 0.0
    elif net > 100:
        return 1.0
    else:
        return 1.0 / (1.0 + math.exp(-net))  # Sigmoid function


def file_net(file_name):
    if SpamFilter.spam_set.__contains__(file_name):
        spam_net = SpamFilter.initial_weight
        if SpamFilter.spam_tokens_dict[file_name] is not None:
            for spam_key in SpamFilter.spam_tokens_dict[file_name].keys():
                spam_net += (SpamFilter.spam_tokens_dict[file_name].get(spam_key) * SpamFilter.weight_matrix.get(spam_key))
        # Calling function to compute the value of the Sigmoid function
        return apply_logistic_regression(spam_net)
    elif SpamFilter.ham_set.__contains__(file_name):
        ham_net = SpamFilter.initial_weight
        if SpamFilter.ham_tokens_dict[file_name] is not None:
            for ham_key in SpamFilter.ham_tokens_dict[file_name].keys():
                ham_net += (SpamFilter.ham_tokens_dict[file_name].get(ham_key) * SpamFilter.weight_matrix.get(ham_key))
            # Calling function to compute the value of the Sigmoid function
        return apply_logistic_regression(ham_net)
    else:
        return 0


def token_frequency(file_name, token):
    if SpamFilter.spam_set.__contains__(file_name):
        for spam_key in SpamFilter.spam_tokens_dict[file_name].keys():
            if spam_key == token:
                return SpamFilter.spam_tokens_dict[file_name].get(spam_key)
            else:
                return 0
    elif SpamFilter.ham_set.__contains__(file_name):
        for ham_key in SpamFilter.ham_tokens_dict[file_name].keys():
            if ham_key == token:
                return SpamFilter.ham_tokens_dict[file_name].get(ham_key)
            else:
                return 0
    else:
        return 0


def train_logistic_regression():
    for token in SpamFilter.tokens:
        # Assigning random weights to tokens
        weight = 2 * (random.randint(0, 100) / 100) - 1
        SpamFilter.weight_matrix[token] = weight
    for i in range(0, SpamFilter.runs):
        for token in SpamFilter.tokens:
            a1 = 0
            for file_name in SpamFilter.file_name_set:
                # Calculating the frequency of occurrence of each token
                freq = token_frequency(file_name, token)

                if SpamFilter.spam_set.__contains__(file_name):
                    target = 1.0  # spam
                else:
                    target = 0.0  # ham

                # Computing the summation of products of weights and token values
                est_opt = file_net(file_name)
                e = (target - est_opt)
                a1 += freq * e

            # Computing the final theta j value for the LR formula
            new_weight = SpamFilter.weight_matrix[token] + SpamFilter.learning_rate * (a1 - (SpamFilter.lambda_value * SpamFilter.weight_matrix[token]))
            SpamFilter.weight_matrix[token] = new_weight


def test_logistic_regression(test_data_dict):
    result = 0.0

    for key in test_data_dict.keys():
        if SpamFilter.weight_matrix.keys().__contains__(key):
            # Computing the sum of products of weights and token values
            result += (test_data_dict[key] * SpamFilter.weight_matrix.get(key))

    # Total value
    result += SpamFilter.initial_weight

    return result > 0


def read_test(mail_path, is_spam):
    count = 0
    mails = [f for f in listdir(mail_path) if isfile(join(mail_path, f))]
    for file in mails:
        test_data_dict = dict()

        for line in read_file(mail_path + "/" + file):
            for word in line.lower().strip().split(" "):
                clean_data = clean_string(word)
                if clean_data is not None and clean_data != "":
                    if test_data_dict.keys().__contains__(clean_data):
                        test_data_dict[clean_data] = test_data_dict.get(clean_data) + 1
                    else:
                        test_data_dict[clean_data] = 1

        if SpamFilter.filter_stop_words:
            for stop_word in SpamFilter.stop_words:
                if test_data_dict.keys().__contains__(stop_word):
                    test_data_dict.pop(stop_word)

        logistic_regression_result = test_logistic_regression(test_data_dict)
        if is_spam and logistic_regression_result:
            count += 1
        elif not is_spam and not logistic_regression_result:
            count += 1
    return count


#######################
# Main Code Execution #
#######################

# Step 1 : Validating input parameters
train_folder = ""
test_folder = ""
# The list of precompiled stopwords have been taken from "http://www.ranks.nl/stopwords"
stop_word_file = "stopwords.txt"

try:
    opts, args = getopt.getopt(sys.argv[1:], "ha:l:r:t:v:s:f:")
except getopt.GetoptError:
    print('-a <learning_rate> -l <lambda_value> -r <runs> -t <train_folder> -v <test_folder> -s <stop_word_file> -f <filter_stop_words>')
    sys.exit(1)
for opt, arg in opts:
    if opt == '-h':
        print('-a <learning_rate> -l <lambda_value> -r <runs> -t <train_folder> -v <test_folder> -s <stop_word_file> -f <filter_stop_words>')
        sys.exit()
    elif opt in "-a":
        # Learning Rate 'alpha'
        SpamFilter.learning_rate = int(arg)
    elif opt in "-l":
        # Lambda Value
        SpamFilter.lambda_value = int(arg)
    elif opt in "-r":
        # Number of iterations
        SpamFilter.runs = int(arg)
    elif opt in "-t":
        train_folder = arg
    elif opt in "-v":
        test_folder = arg
    elif opt in "-s":
        stop_word_file = arg
    elif opt in "-p":
        if arg == "true" or arg == "True" or arg == "1":
            SpamFilter.filter_stop_words = True
        else:
            SpamFilter.filter_stop_words = False

# Step 2 : Creating Files from CMD arguments
training_spam = (train_folder + "/train/spam")
training_ham = (train_folder + "/train/ham")

test_spam = (test_folder + "/test/spam")
test_ham = (test_folder + "/test/ham")

# Step 3 : Calling the read() function with spam and ham folders as inputs
read(training_spam)
read(training_ham)

# Step 4 : Filter stopwords
if SpamFilter.filter_stop_words:
    print("Filtering stopwords")
    SpamFilter.stop_words = set(read_file(stop_word_file))
    for data in SpamFilter.stop_words:
        if SpamFilter.tokens.__contains__(data):
            SpamFilter.tokens.remove(data)

# Step 5 : Find and count tokens
find_tokens(training_spam, True)
find_tokens(training_ham, False)

# Step 6 : Training System
train_logistic_regression()

# Step 7 : Reading Test Data and evaluating
correctS = read_test(test_spam, True)
correctH = read_test(test_ham, False)

# Step 8 : Calculating the accuracy
total = [f for f in listdir(training_spam) if isfile(join(training_spam, f))].__len__() \
        + [f for f in listdir(training_ham) if isfile(join(training_ham, f))].__len__()
accuracy = (correctH + correctS) / total
print("The accuracy of Spam Filter using Logistic Regression: " + accuracy.__str__())
