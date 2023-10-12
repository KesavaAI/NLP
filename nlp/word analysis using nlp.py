import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
text = "this is me:kesav reddy"
words = word_tokenize(text)
# Use NLTK&#39;s part-of-speech (POS) tagger to get the morphological features of each word
from nltk import pos_tag
pos_tags = pos_tag(words)
print(pos_tags)