# rename 'X' and 'Y' to avoid confusion
original_df.rename(index=str, columns={"X": "x_coord", "Y": "y_coord"}, inplace=True)

# In http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.names,
# Missing Attribute Values: None, but better safe than sorry
import numpy as np

print('Number of rows before removing rows with missing values: ' + str(original_df.shape[0]))

# Replace ? with np.NaN
original_df.replace('?', np.NaN, inplace=True)

# Remove rows with np.NaN
original_df.dropna(how='any', inplace=True)

print('Number of rows after removing rows with missing values: ' + str(original_df.shape[0]))
