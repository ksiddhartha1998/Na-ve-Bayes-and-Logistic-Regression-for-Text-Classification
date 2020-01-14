import sys
import getopt
import math
import re
from os import listdir
from os.path import isfile, join


class SpamFilter:
    # Variables
    filter_stop_words = True
    # Prior Probabilities
    prior_probability_spam = 0
    prior_probability_ham = 0
    # Counters for spam and ham tokens
    total_spam_count = 0
    total_ham_count = 0
    # Stop words and total number of tokens
    stopwords = set()
    tokens = set()
    # Data structures for storing spam and ham tokens
    spam_tokens = dict()
    ham_tokens = dict()
    # Data structure for storing conditional probabilities
    conditional_probability_spam = dict()
    conditional_probability_ham = dict()


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
        file = mail_path + "/" + file
        for line in read_file(file):
            for word in line.lower().strip().split(" "):
                clean_data = clean_string(word)
                if clean_data is not None and clean_data != "":
                    if SpamFilter.tokens.__contains__(clean_data):
                        if is_spam:
                            SpamFilter.total_spam_count += 1
                            if SpamFilter.spam_tokens.keys().__contains__(clean_data):
                                SpamFilter.spam_tokens[clean_data] = SpamFilter.spam_tokens.get(clean_data) + 1
                            else:
                                SpamFilter.spam_tokens[clean_data] = 1
                        else:
                            SpamFilter.total_ham_count += 1
                            if SpamFilter.ham_tokens.keys().__contains__(clean_data):
                                SpamFilter.ham_tokens[clean_data] = SpamFilter.ham_tokens.get(clean_data) + 1
                            else:
                                SpamFilter.ham_tokens[clean_data] = 1


def train_multinomial_naive_bayes(training_spam_mails, training_ham_mails):
    spam_count = [f for f in listdir(training_spam_mails) if isfile(join(training_spam_mails, f))].__len__()
    ham_count = [f for f in listdir(training_ham_mails) if isfile(join(training_ham_mails, f))].__len__()
    s = 1.0 * spam_count / (spam_count + ham_count)
    h = 1.0 - s

    SpamFilter.prior_probability_spam = math.log(s)
    SpamFilter.prior_probability_ham = math.log(h)

    for data in SpamFilter.tokens:
        if SpamFilter.spam_tokens.keys().__contains__(data):
            spam_probability = (SpamFilter.spam_tokens.get(data) + 1.0) / (SpamFilter.total_spam_count + SpamFilter.tokens.__len__() + 1.0)
            SpamFilter.conditional_probability_spam[data] = math.log(spam_probability)
        if SpamFilter.ham_tokens.keys().__contains__(data):
            ham_probability = (SpamFilter.ham_tokens.get(data) + 1.0) / (SpamFilter.total_ham_count + SpamFilter.tokens.__len__() + 1.0)
            SpamFilter.conditional_probability_ham[data] = math.log(ham_probability)


def apply_multinomial_naive_bayes(file):
    sa = 0.0
    ha = 0.0
    for line in read_file(file):
        for word in line.lower().strip().split(" "):
            clean_data = clean_string(word)
            if clean_data is not None and clean_data != "":
                if SpamFilter.filter_stop_words:
                    if not SpamFilter.stopwords.__contains__(clean_data):
                        if SpamFilter.conditional_probability_spam.keys().__contains__(clean_data):
                            sa += SpamFilter.conditional_probability_spam[clean_data]
                        else:
                            sa += math.log(1.0 / (SpamFilter.total_spam_count + SpamFilter.tokens.__len__() + 1.0))
                        if SpamFilter.conditional_probability_ham.keys().__contains__(clean_data):
                            ha += SpamFilter.conditional_probability_ham[clean_data]
                        else:
                            ha += math.log(1.0 / (SpamFilter.total_ham_count + SpamFilter.tokens.__len__() + 1.0))
                else:
                    if SpamFilter.conditional_probability_spam.keys().__contains__(clean_data):
                        sa += SpamFilter.conditional_probability_spam[clean_data]
                    else:
                        sa += math.log(1.0 / (SpamFilter.total_spam_count + SpamFilter.tokens.__len__() + 1.0))
                    if SpamFilter.conditional_probability_ham.keys().__contains__(clean_data):
                        ha += SpamFilter.conditional_probability_ham[clean_data]
                    else:
                        ha += math.log(1.0 / (SpamFilter.total_ham_count + SpamFilter.tokens.__len__() + 1.0))
    # Spam probability
    sa += SpamFilter.prior_probability_spam
    # Ham probability
    ha += SpamFilter.prior_probability_ham

    return sa > ha


def read_test(mail_path, is_spam):
    count = 0.0
    mails = [f for f in listdir(mail_path) if isfile(join(mail_path, f))]
    for file in mails:
        file = mail_path + "/" + file
        multinomial_naive_bayes_result = apply_multinomial_naive_bayes(file)
        if is_spam and multinomial_naive_bayes_result:
            count += 1
        elif not is_spam and not multinomial_naive_bayes_result:
            count += 1
    return count


#######################
# Main Code Execution #
#######################

# Step 1 : Reading and Validating input parameters
train_folder = ""
test_folder = ""
# The list of precompiled stopwords have been taken from "http://www.ranks.nl/stopwords"
stop_word_file = "stopwords.txt"

try:
    opts, args = getopt.getopt(sys.argv[1:], "ht:v:s:f:")
except getopt.GetoptError:
    print('-t <train_folder> -v <test_folder> -s <stop_word_file> -f <filter_stop_words>')
    sys.exit(1)
for opt, arg in opts:
    if opt == '-h':
        print('-t <train_folder> -v <test_folder> -s <stop_word_file> -f <filter_stop_words>')
        sys.exit()
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
    SpamFilter.stopwords = set(read_file(stop_word_file))
    for data in SpamFilter.stopwords:
        if SpamFilter.tokens.__contains__(data):
            SpamFilter.tokens.remove(data)

# Step 5 : Find and count tokens
find_tokens(training_spam, True)
find_tokens(training_ham, False)

# Step 6 : Training System
train_multinomial_naive_bayes(training_spam, training_ham)

# Step 7 : Reading Test Data and evaluating
correctS = read_test(test_spam, True)
correctH = read_test(test_ham, False)

# Step 8 : Calculating the accuracy
total = [f for f in listdir(training_spam) if isfile(join(training_spam, f))].__len__() \
        + [f for f in listdir(training_ham) if isfile(join(training_ham, f))].__len__()
accuracy = (correctH + correctS) / total
print("The accuracy of Spam Filter using Multinomial Naive Bayes: " + accuracy.__str__())
