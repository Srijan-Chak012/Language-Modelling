# Create a corpus of words
def create_ngrams(corpus, n):
    ngrams = {}
    for i in range(len(corpus) - (n-1)):
        ngram = tuple(corpus[i:i+n])
        if ngram in ngrams:
            ngrams[ngram] += 1
        else:
            ngrams[ngram] = 1
    return ngrams

text = ["My", "name", "is", "srijan"]

unigram = create_ngrams(text, 1)
bigram = create_ngrams(text, 2)
trigram = create_ngrams(text, 3)
fourgram = create_ngrams(text, 4)

all_ngrams = [unigram, bigram, trigram, fourgram]

print(all_ngrams)

# Get unique ngrams

def get_unique_ngrams(ngrams):
    unique_ngrams = set(ngrams.keys())
    return unique_ngrams

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

def kneser_ney_smoothing(ngrams, n, d=0.75):
    smoothed_ngrams = {}
    for ngram in ngrams:
        if n == 2:
            word = ngram[-1]
            count = 0
            for x, y in all_ngrams[1].items():
                if y == word:
                    count += 1
            denom = len(all_ngrams[1])
        else:
            denom_seq = ngram[:-1]
            denom = all_ngrams[n-2].get(denom_seq, 0)
        
        first_term = max(ngrams[ngram] - d, 0)
        second_term = d * sum(kneser_ney_smoothing(history, n-1).get(history[:-1], 0) for history in ngrams if history[1:] == ngram[:-1])
        smoothed_prob = (first_term + second_term) / denom
        smoothed_ngrams[ngram] = smoothed_prob
    return smoothed_ngrams

sequence = ("My", "name", "is", "Darcy")
smoothed_ngrams = kneser_ney_smoothing(all_ngrams[3], 4)
print(smoothed_ngrams.get(sequence, 0))

