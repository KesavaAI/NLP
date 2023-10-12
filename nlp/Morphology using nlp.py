# Define the prefixes and suffixes to analyze
prefixes = ["un","in","im","dis"]
suffixes = ["able","ness","ful","ment"]
def generate_forms(word):
# Start with the original word
  forms = [word]
# Remove any prefixes
  for prefix in prefixes:
    if word.startswith(prefix):
      forms.append(word[len(prefix):])
# Remove any suffixes
  for suffix in suffixes:
    if word.endswith(suffix):
      forms.append(word[:-len(suffix)])
# Remove both prefixes and suffixes
  for prefix in prefixes:
    for suffix in suffixes:
      if word.startswith(prefix) and word.endswith(suffix):
        forms.append(word[len(prefix):-len(suffix)])
  return forms
# Example usage
word = "unhappiness"
forms = generate_forms(word)
print(f"All possible morphological forms of '{word}':")
for form in forms:
  print(form)