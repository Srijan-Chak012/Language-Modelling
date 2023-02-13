import re
import random

file_pride = open("corpus/Pride and Prejudice - Jane Austen.txt", "r")
file_pride_text = file_pride.read()

# print(file_pride_text)

file_ulysses = open("corpus/Ulysses - James Joyce.txt", "r")
file_ulysses_text = file_ulysses.read()

# print(file_ulysses_text)

def url_match(m):
    return '<URL>'

# Regex for hashtag

def hashtag_match(m):
    return '<HASHTAG>'

# Regex for @

def mention_match(m):
    return '<MENTION>'

# Regex for punctuation
punc_regex = r"[!\"#\$%&\'\(\)\*\+,-\./:;=\?\[\\\]\^_`{\|}~]"

# Tokenize text into sentences

def sentence_tokeniser(text):
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
    # remove spaces at the beginning and end of each sentence
    sentences = [s.strip() for s in sentences]
    # remove empty sentences
    sentences = [s for s in sentences if s]
    return sentences

# Tokenize text into words

def tokeniser(text):
    text = text.lower()
    http_url_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.IGNORECASE)
    # Regex for URLS without HTTP or HTTPS
    url_regex = re.compile(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.IGNORECASE)
    hashtag_regex = re.compile(r'#\w+', re.IGNORECASE)
    mention_regex = re.compile(r'@\w+', re.IGNORECASE)
    punc_regex = r"[!\"#\$%&\'\(\)\*\+,-\./:;=\?\[\\\]\^_`{\|}~]"
    new_text = http_url_regex.sub(url_match, text)
    new_text = url_regex.sub(url_match, new_text)
    new_text = hashtag_regex.sub(hashtag_match, new_text)
    new_text = mention_regex.sub(mention_match, new_text)
    new_text = re.sub(punc_regex, "", new_text, 0, re.MULTILINE)
    words = new_text.split()
    return words

pride_sentences = sentence_tokeniser(file_pride_text)
pride_words = tokeniser(file_pride_text)
ulysses_words = tokeniser(file_ulysses_text)

f = open("prideoutput.txt", "w")
f.write(", ".join(pride_words))
f.close()

f = open("ulyssesoutput.txt", "w")
f.write(", ".join(ulysses_words))
f.close()

f = open("pride_sentences.txt", "w")
f.write("\n ".join(pride_sentences))
f.close()
# Create a corpus of words
def create_ngrams(corpus, n):
    ngrams = {}
    for i in range(len(corpus) - (n-1)):
        ngram = tuple(corpus[i:i+n])
        if ngram in ngrams:
            ngrams[ngram] += 1
        else:
            ngrams[ngram] = 1
        # print(ngram)
    return ngrams

unigram = create_ngrams(pride_words, 1)
def clean_text(text):
    # if unigram probability is less than or equal to one, replace the word in original text with <UNK>
    for word in text:
        if unigram.get((word,), 0) <= 1:
            text[text.index(word)] = "<UNK>"

    return text

pride_words = clean_text(pride_words)
f = open("prideoutput2.txt", "w")
f.write(", ".join(pride_words))
f.close()

unigram = create_ngrams(pride_words, 1)
f = open("unigram.txt", "w")
for key, value in unigram.items():
    f.write(str(key) + " " + str(value))
    f.write(" ")
f.close()

bigram = create_ngrams(pride_words, 2)
trigram = create_ngrams(pride_words, 3)
fourgram = create_ngrams(pride_words, 4)

all_ngrams = [unigram, bigram, trigram, fourgram]

# print(all_ngrams)

def backoff(ngrams, n):
    if n == 2:
        word = ngrams[-1:0]
        # Number of bigrams with the word as the last element
        count = 0
        print(word)
        for x, y in all_ngrams[1].keys():
            if y == word:
                count += 1
        
        print(count)
        return count/len(all_ngrams[1])
         
    else:
        # ignore the first elements and get n-1 grams
        n_minus_1_gram = ngrams[1:]
        # get the count of n-1 grams
        n_minus_1_gram_count = all_ngrams[n-2][n_minus_1_gram]
        print(n_minus_1_gram)

        if n_minus_1_gram_count == 0:
            return backoff(n_minus_1_gram, n-1)

        else:
            return all_ngrams[n-1][ngrams]

def kneser_ney_smoothing(sequence, n, d=0.75):
    if n == 2:
        context = sequence[0]
        word = sequence[1]
        first_denom = 0
        print(context, word)
        count = 0
        for x, y in all_ngrams[1].keys():
            if x == context:
                first_denom += 1
            if y == word:
                count += 1
        denom = len(all_ngrams[1])
        print(sequence[0])
        if first_denom == 0:
            return 0
        print("first denom ", first_denom)
        # print(all_ngrams[n-1].get(sequence, 0))
        seq = (sequence[:-1])
        first_denom = max(all_ngrams[n-2].get(seq, 0) - d, 0)
        first_term = max(all_ngrams[n-1].get(sequence, 0) - d, 0)/first_denom
        print("first ", first_term, sequence)
        second_term = d * count / denom
        print("second ", second_term, sequence)
        smoothed_prob = (first_term + second_term)
        return smoothed_prob
    else:
        denom_seq = sequence[:-1]
        denom = all_ngrams[n-2].get(denom_seq, 0)
        # print(denom)
        if denom == 0:
            # print("Here")
            return 1e6
        # print(all_ngrams[n-1].get(sequence, 0))
        first_term = max(all_ngrams[n-1].get(sequence, 0) - d, 0)/denom
        print("first ", first_term, sequence)

        # get unique continuations of n-1 grams
        unique_continuations = set()
        for x in all_ngrams[n-1].keys():
            if x[:-1] == denom_seq:
                unique_continuations.add(x[-1])
        second_term_count = len(unique_continuations)
        second_term = d * second_term_count * kneser_ney_smoothing(sequence[1:], n-1) / denom
        print("second ", second_term, sequence)
        smoothed_prob = (first_term + second_term)
        return smoothed_prob

# sequence = ["between", "him", "and", "darcy"]
# sequence = clean_text(sequence)
# sequence = tuple(sequence)
# print(sequence)
# prob = kneser_ney_smoothing(sequence, 4)
# print(prob)

def witten_bell_smoothing(sequence, n):
    if n == 1:
        all_words = sum(all_ngrams[0].values())
        count = all_ngrams[0].get(sequence, 0)/all_words
        print(all_ngrams[0].get(sequence, 0))
        print("basecount ", count, sequence)
        return count
    history = sequence[:-1]
    count_history = all_ngrams[n-2].get(history, 0)
    if count_history == 0:
        return 0
    count_term = all_ngrams[n-1].get(sequence, 0)
    p_ml = count_term/count_history

    # get unique continuations of n-1 grams
    unique_continuations = set()
    for x in all_ngrams[n-1].keys():
        if x[:-1] == history:
            unique_continuations.add(x[-1])
    # print(unique_continuations)

    # lamba term calculation
    n_term_num = len(unique_continuations)
    n_term_denom = count_history + n_term_num
    n_term = n_term_num/n_term_denom

    lambda_term = 1 - n_term 
    print("lambda ", lambda_term, sequence)
    
    first_term = lambda_term * p_ml
    print("first ", first_term, sequence)

    context = sequence[1:]
    second_term = n_term * witten_bell_smoothing(context, n-1)
    print("second ", second_term, sequence)
    p_wb = first_term + second_term
    return p_wb

sequence = ["agar", "mai", "kahoon", "ye"]
sequence = clean_text(sequence)
sequence = tuple(sequence)
print(sequence)
prob = witten_bell_smoothing(sequence, 4)
print(prob)

def create_testing_dataset(text):   
    text_sentences = sentence_tokeniser(text)

    # Take 1000 random sentences and split the dataset
    test_data = random.sample(text_sentences, 1000)
    
