def generate_word_form(root, suffix):
    """This function takes a root and a suffix as input and generates a word form by appending the suffix to the root. It returns the generated word form."""
    return root + suffix

# Example usage:
root = "play"
suffix = "er"
word_form = generate_word_form(root, suffix)
print(word_form)

root = "book"
suffix = "ed"
word_form = generate_word_form(root, suffix)
print(word_form)
