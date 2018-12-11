# Specify the name of the target
target = 'area'

# Get the target vector
y = df[target].values

# Specify the name of the features
features = list(df.drop(target, axis=1).columns)

# Get the feature vector
X = df[features].values

from sklearn.preprocessing import LabelEncoder

# Declare the LabelEncoder
class_le = LabelEncoder()

# Enclode the target
y = class_le.fit_transform(y)
