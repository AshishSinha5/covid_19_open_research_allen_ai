from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

file_path = 'data/initial_corpus.txt'

num_sent = 0
with open(file_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        num_sent += len(sent_tokenize(line))

num_words = []
num_token = []
w = []
with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(tqdm(f)):
        w += word_tokenize(line)
        if i == 100:
            v = list(set(w))
            num_words.append(len(w))
            num_token.append(len(v))
        elif i == 1000:
            v = list(set(w))
            num_words.append(len(w))
            num_token.append(len(v))
        elif i == 5000:
            v = list(set(w))
            num_words.append(len(w))
            num_token.append(len(v))
        elif i == 10000:
            v = list(set(w))
            num_words.append(len(w))
            num_token.append(len(v))
        elif i == 20000:
            v = list(set(w))
            num_words.append(len(w))
            num_token.append(len(v))
        elif i == 25000:
            v = list(set(w))
            num_words.append(len(w))
            num_token.append(len(v))
        elif i == 45000:
            v = list(set(w))
            num_words.append(len(w))
            num_token.append(len(v))
        elif i == 50000:
            v = list(set(w))
            num_words.append(len(w))
            num_token.append(len(v))

total_words = len(w)
vocab = list(set(w))
vocab_size = len(vocab)

print("Number of Sentences  = {}".format(num_sent))
print("Number of words  = {}".format(total_words))
print("Vocabulary Size = {}".format(vocab_size))

v1, n1 = num_token[4], num_words[4]
v2, n2 = num_token[6], num_words[6]

beta = (np.log(v2) - np.log(v1)) / (np.log(n2) - np.log(n1))
K = v1 / (n1 ** beta)

print("K = {} \nbeta = {}".format(K, beta))

x = np.linspace(350000, num_words[-1] + 1000, 10)
y = K * (x ** beta)

plt.title('Heaps\' Law Verification on Covid Corpus')
plt.scatter(num_words, num_token, label='Actual')
plt.scatter(x, y, label='Estimated by Heaps\' Law')
plt.xlabel('Text Size')
plt.ylabel('Vacab Size')
plt.legend()
plt.savefig('plots/heaps_law.png')
plt.show()
plt.close()
