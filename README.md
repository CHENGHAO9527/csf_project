# Fianl Project for CSF 

## Group Member: Chen Miao; Jiajun Wu; Hao Chen

Buildingthe  package to get Forest Fires Dataset ready for machine learning modeling for different frameworks.
Datasets: http://archive.ics.uci.edu/ml/datasets/Forest+Fires

We use Python to analyze the forest fire data. Specific packages used are Numpy, Pandas, and Scikit-learn. We use Matplotlib for visualizations. For the methods the team used to make our predictor, we tried Neural Net, Random Forest and Support Vector Machine (SVM).

---
### Feature:

* data_preprocessing.py
* get_feature.py
* neural_network.py
* SVM_model.py
* Random_Forest_Model.py

---
### Sample Code:

from sklearn.ensemble import RandomForestClassifier
From Forest_Files_Prediction.data_preprocessing import ffp_csf
From Forest_Files_Prediction.get_feature import ffp_get_feature,ffp_train_test_split
from matplotlib.colors import ListedColormap

---
### output:

    # 2 categories:
    # Misclassified samples: 84
    # The accuracy of random forest is: 0.46153846153846156

    # 3 categories:
    # Misclassified samples: 90
    # The accuracy of random forest is: 0.4230769230769231
