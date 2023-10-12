import nltk
nltk.download('brown')
from nltk.corpus import brown

def train_and_evaluate_tagger(train_sents, test_sents, context_size):
    # Train a tagger on the training set
    if context_size == 0:
        # Unigram tagger
        tagger = nltk.UnigramTagger(train_sents)
    elif context_size == 1:
        # Bigram tagger
        tagger = nltk.BigramTagger(train_sents)
    else:
        # Trigram tagger
        tagger = nltk.TrigramTagger(train_sents)

    # Check if the test set is empty
    if len(test_sents) == 0:
        return 0.0

    # Evaluate the performance of the tagger on the test set
    accuracy = tagger.evaluate(test_sents)

    return accuracy

# Load the Brown corpus
brown_sents = brown.tagged_sents()

# Split the corpus into a training set and a test set
train_sents = brown_sents[:4000]
test_sents = brown_sents[4000:]

# Test the performance of different context sizes on a small training corpus
print("Performance on small training corpus (500 sentences):")
print("Context size 0 (unigram):", train_and_evaluate_tagger(train_sents, test_sents, 0))
print("Context size 1 (bigram):", train_and_evaluate_tagger(train_sents, test_sents, 1))
print("Context size 2 (trigram):", train_and_evaluate_tagger(train_sents, test_sents, 2))

# Test the performance of different context sizes on the full training corpus
print("Performance on full training corpus (4000 sentences):")
print("Context size 0 (unigram):", train_and_evaluate_tagger(train_sents, [], 0))
print("Context size 1 (bigram):", train_and_evaluate_tagger(train_sents, [], 1))
print("Context size 2 (trigram):", train_and_evaluate_tagger(train_sents, [], 2))
