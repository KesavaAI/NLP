import numpy as np
# Define the sparse bigram table
table = np.array([[0, 1, 0, 0],
[1, 0, 0, 0],
[0, 0, 0, 1],
[0, 0, 1, 0]])
# Apply add-one smoothing
smoothed_table = (table + 1) / (table.sum(axis=1, keepdims=True) +
table.shape[1])
# Print the original and smoothed tables
print("Original table:")
print(table)
print("\nSmoothed table:")
print(smoothed_table)