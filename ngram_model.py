from tqdm import tqdm
import re
import collections
from nltk import ngrams
import operator
import random
import math

from nltk.tokenize import sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
import gc
import in_place


# Preprocessing Steps

def remove_spaces(file_path):
    """
    args: file_path - file path of the corpus
    returns: converts text to lower case and removes spaces
    """
    print('---------------------------------------------------------')
    print("removing extra spaces")
    with in_place.InPlace(file_path, encoding = 'utf-8') as file:
        for line in tqdm(file):
            line = re.sub(r'\s+\.', '. ', line)
            line = re.sub(r'(\.)(\w+)',r'\1 \2', line)
            file.write(line)
    gc.collect()
    print('Done')
    print('---------------------------------------------------------')


def sentence_tokenize(file_path):
    """
    args: file_path - file path of text corpus
    returns: writes one sentence per line
    """
    print('---------------------------------------------------------')
    print("sentence tokenizing")
    pat = re.compile(r'([A-Z][^\.!?]*[\.!?])', re.M)
    with open('data/preprocessed.txt', 'w', encoding='utf-8') as f2:
        with open(file_path, 'r', encoding='utf-8') as f1:
            for i, line in tqdm(enumerate(f1)):
                # print(line)
                sentences = sent_tokenize(line)
                # print(sentences)
                for sentence in sentences:
                    f2.write(sentence)
                    f2.write('\n')

    print('Done')
    print('---------------------------------------------------------')

def to_lower(file_path):
    """
    args: file_path - file path of the corpus
    returns: converts text to lower case, removes sentences with less than 50 tokens
    """
    print('---------------------------------------------------------')
    print("converting to lower case and adding delimiters")
    with in_place.InPlace(file_path, encoding = 'utf-8') as file:
        for line in tqdm(file):
            if len(line) > 1:
                line = line.lower()
                file.write(line[:-2])
                file.write('\n')
    gc.collect()
    print('Done')
    print('---------------------------------------------------------')
    with in_place.InPlace(file_path, encoding = 'utf-8') as file:
        for line in tqdm(file):
            if len(line) > 50:
                file.write(line)
                #file.write('\n')
    gc.collect()

# The corpus containes more that 83L sentences due to computaional limitations I only use portion of data to build the
# language model


def split_train_test(file_path, corpus_prop=0.5, train_prop=0.9):
    """
    args : file_path : path of file for preprocessed corpus
    returns : splits raw text to train and test set
    """

    """
    1 - count the number of lines
    2 - |train_set| = train_prop * number_of_lines
    3 - |test_set| = (1 - train_prop) * number_lines
    """

    train_file = 'data/train_preprocessed.txt'
    test_file = 'data/test_preprocessed.txt'
    print('getting line count')
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, l in tqdm(enumerate(f)):
            pass
    num_lines = i + 1
    print('Number of lines = {}'.format(num_lines))
    print('building train, test set using only {} % of the entire corpus'.format(corpus_prop * 100))
    f1 = open(train_file, 'w', encoding='utf-8')
    f2 = open(test_file, 'w', encoding='utf-8')
    k = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, l in tqdm(enumerate(f)):
            if i < (num_lines * corpus_prop):
                k += 1
                if k < (num_lines * corpus_prop * train_prop):
                    f1.write(l)
                else:
                    f2.write(l)
    with open(train_file, 'r', encoding='utf-8') as f:
        for i, l in tqdm(enumerate(f)):
            pass
    print("Number of sentences in train set = {}".format(i + 1))
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, l in tqdm(enumerate(f)):
            pass
    print("Number of sentences in test set = {}".format(i + 1))

# Trigram Model

def build_trigram_model(file_path):
    trigram_model = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    with open(file_path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        print('building trigram frequency dictionary')
        for l in tqdm(lines):
            for w1, w2, w3 in ngrams(l.split(' '),3, pad_right=True, pad_left=True):
                trigram_model[(w1, w2)][w3] += 1
        print('Normalizing frequncies to create probability distribution')
        for w1_w2 in tqdm(trigram_model):
            total_count = float(sum(trigram_model[w1_w2].values()))
            for w3 in trigram_model[w1_w2]:
                trigram_model[w1_w2][w3] /= total_count
    return trigram_model

# Fourgram Model

def build_fourgram_model(file_path):
    fourgram_model = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    with open(file_path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        print('building fourgram frequency dictionary')
        for i, l in tqdm(enumerate(lines)):
            for w1, w2, w3, w4 in ngrams(l.split(' '), 4, pad_right=True, pad_left=True):
                fourgram_model[(w1, w2, w3)][w4] += 1
            #if i == 100:
            #    break
        # print(list(fourgram_model.keys())[0])
        print('Normalizing frequncies to create probability distribution')
        for w1_w2_w3 in fourgram_model:
            total_count = float(sum(fourgram_model[w1_w2_w3].values()))
            for w4 in fourgram_model[w1_w2_w3]:
                fourgram_model[w1_w2_w3][w4] /= total_count
    return fourgram_model



if __name__ == "__main__":
    file_path = 'data/initial_corpus.txt'
    remove_spaces(file_path)
    sentence_tokenize(file_path)

    processed_corpus_path = 'data/preprocessed.txt'
    to_lower(file_path)
    split_train_test(processed_corpus_path, train_prop=0.8)

    train_file = 'data/train_preprocessed.txt'
    trigram_model = build_trigram_model(train_file)

    most_frequent_word = max(trigram_model[None, None].items(), key=operator.itemgetter(1))[0]

    text = [None, most_frequent_word]
    sentence_finished = False
    while not sentence_finished:
        # select a random probability threshold
        r = random.random()
        accumulator = .0

        for word in trigram_model[tuple(text[-2:])].keys():
            accumulator += trigram_model[tuple(text[-2:])][word]
            # select words that are above the probability threshold
            if accumulator >= r:
                text.append(word)
                break

        if text[-2:] == [None, None]:
            sentence_finished = True

    print('Randomly generated sentence from the model')
    print(' '.join([t for t in text if t]))


    fourgram_model = build_fourgram_model(train_file)
    most_frequent_word_4gram = max(fourgram_model[None, None, None].items(), key=operator.itemgetter(1))[0]

    text = [None, None, most_frequent_word_4gram]
    sentence_finished = False
    while not sentence_finished:
        # select a random probability threshold
        r = random.random()
        accumulator = .0

        for word in fourgram_model[tuple(text[-3:])].keys():
            accumulator += fourgram_model[tuple(text[-3:])][word]
            # select words that are above the probability threshold
            if accumulator >= r:
                text.append(word)
                break

        if text[-3:] == [None, None, None]:
            sentence_finished = True

    print('Randomly generated sentence from the 4gram model')
    print(' '.join([t for t in text if t]))

    ## Model Evaluation

    with open('data/train_preprocessed.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        tokenized_text = [list(line[:-1].split(' ')) for line in lines]

    n = 3
    train_data_trigrams, padded_vocab = padded_everygram_pipeline(n, tokenized_text)
    trigram_model = MLE(n)
    trigram_model.fit(train_data_trigrams, padded_vocab)

    with open('data/test_preprocessed.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        tokenized_text_test = [list(line[:-1].split(' ')) for i, line in enumerate(lines) if (i < 50000)]

    test_data_trigrams, _ = padded_everygram_pipeline(n, tokenized_text_test)

    perplexity = 0
    n_valid = 0
    for i, test in tqdm(enumerate(test_data_trigrams)):
        try:
            p = trigram_model.perplexity(test)
            if p > 0 and (not math.isinf(p)):
                perplexity += p
                n_valid += 1
        except ZeroDevideError:
            continue


    print("Average perplexity of tri gram model over 50000 validation sentences = {}".format(perplexity/n_valid))

    n1 = 4
    train_data_fourgrams, padded_vocab = padded_everygram_pipeline(n1, tokenized_text)
    fourgram_model = MLE(n1)
    fourgram_model.fit(train_data_fourgrams, padded_vocab)

    test_data_fourgrams, _ = padded_everygram_pipeline(n1, tokenized_text_test)

    perplexity = 0
    n_valid = 0
    for i, test in tqdm(enumerate(test_data_fourgrams)):
        try:
            p = fourgram_model.perplexity(test)
            if p > 0 and (not math.isinf(p)):
                perplexity += p
                n_valid += 1
        except ZeroDevideError:
            continue

    print("Average perplexity of four gram model over 50000 validation sentences = {}".format(perplexity / n_valid))