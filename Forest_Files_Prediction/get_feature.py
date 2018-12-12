From Forest_Files_Prediction import data_preprocessing.py
# Specify the name of the target and features
target = 'area'
features = ['temp','RH', 'wind', 'rain', 'FFMC','DMC','DC','ISI']

### 2 categories
# Get the target vector
y2 = df2cats[target].values
print('Two Categories: ')
print('Unencoded class labels:', np.unique(y2))

# Get the feature vector
X2 = df2cats[features].values

from sklearn.preprocessing import LabelEncoder

# Declare the LabelEncoder
class_le = LabelEncoder()

# Encode the target
y2 = class_le.fit_transform(y2)
print('Encoded class labels:', np.unique(y2))

### 3 categories
# Get the target vector
y3 = df3cats[target].values
print('')
print('Three Categories: ')
print('Unencoded class labels:', np.unique(y2))

# Get the feature vector
X3 = df3cats[features].values

from sklearn.preprocessing import LabelEncoder

# Declare the LabelEncoder
class_le = LabelEncoder()

# Encode the target
y3 = class_le.fit_transform(y3)

print('Encoded class labels:', np.unique(y3))
