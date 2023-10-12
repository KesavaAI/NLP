import nltk
from nltk import FreqDist
# Sample corpus
corpus = "The quick brown fox jumps over the lazy dog. The lazy dogsleeps all day.";
# Tokenize the corpus into words
words = nltk.word_tokenize(corpus)
# Generate bigrams from the words
bigrams = list(nltk.bigrams(words))
# Print the bigrams
print(bigrams)
# Calculate the frequency of each bigram
freq_dist = FreqDist(bigrams)
# Estimate the probability of each word given its preceding word
probabilities = {}
for bigram in freq_dist:
  prev_word, word = bigram
  if prev_word not in probabilities:
    probabilities[prev_word] = {}
  denominator = freq_dist[prev_word]
  if denominator == 0:
    denominator = 1e-10 # Assign a small default probability if denominator is zero
  probabilities[prev_word][word] = freq_dist[bigram] / denominator
# Calculate the probability of the sentence
sentence = "The lazy dog jumps over the quick brown fox"
sentence_words = nltk.word_tokenize(sentence)
sentence_bigrams = list(nltk.bigrams(sentence_words))
prob_sentence = 1
for bigram in sentence_bigrams:
  prev_word, word = bigram
  if prev_word in probabilities and word in probabilities[prev_word]:
    prob_sentence *= probabilities[prev_word][word]
print("Probability of sentence:", prob_sentence)