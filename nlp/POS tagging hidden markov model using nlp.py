from collections import defaultdict
import numpy as np
# Sample training data
training_data = [
("The", "DT"), ("quick","JJ"), ("brown", "JJ"), ("fox","NN"), ("jumps","VBZ"), (".", ".")
]
# Create set of all possible tags
tags = set(tag for word, tag in training_data)
# Calculate tag counts
tag_counts = defaultdict(int)
for word, tag in training_data:
  tag_counts[tag] += 1
# Calculate transition probabilities
transition_counts = defaultdict(lambda: defaultdict(int))
for i in range(len(training_data)-1):
  current_tag, next_tag = training_data[i][1], training_data[i+1][1]
  transition_counts[current_tag][next_tag] += 1
transition_probs = defaultdict(lambda: defaultdict(int))
for current_tag, next_tags in transition_counts.items():
  total = sum(next_tags.values())
  for next_tag, count in next_tags.items():
    transition_probs[current_tag][next_tag] = count / total
# Calculate emission probabilities
emission_counts = defaultdict(lambda: defaultdict(int))
for word, tag in training_data:
  emission_counts[tag][word] += 1
emission_probs = defaultdict(lambda: defaultdict(int))
for tag, words in emission_counts.items():
  total = sum(words.values())
  for word, count in words.items():
    emission_probs[tag][word] = count / total
print("Transition probabilities: ")
for current_tag, next_tags in transition_probs.items():
  for next_tag, prob in next_tags.items():
     print(f"P({next_tag}|{current_tag}) = {prob:.2f}")
print("\nEmission probabilities:")
for tag, words in emission_probs.items():
  for word, prob in words.items():
    print(f"P({word}|{tag}) = {prob:.2f}")