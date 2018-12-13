# Final Project for Computer Science Foundation 

## Group Member: Cheng Miao; Jiajun Wu; Hao Cheng

Building the  package to get Forest Fires Dataset ready for machine learning modeling for different frameworks.
Datasets: http://archive.ics.uci.edu/ml/datasets/Forest+Fires

We use Python to analyze the forest fire data. Specific packages used are Numpy, Pandas, Scikit-learn，etc. We use Matplotlib for visualizations. For the methods the team used to make our predictor, we tried to implement Neural Network(MLP), Random Forest and Support Vector Machine (SVM).

---
### Feature:

* data_preprocessing.py
* get_feature.py
* neural_network.py
* SVM_model.py
* Random_Forest_Model.py

---
### Sample Code:

    X3_combined = np.vstack((X3_train_pca, X3_test_pca))
    y3_combined = np.hstack((y3_train, y3_test))
    test_start3 = X3_train_pca.shape[0]
    test_end3 = X3_train_pca.shape[0] + X3_test_pca.shape[0]


---
### Installation
Type the following commands in your terminal.
```
git clone https://github.com/CHENGHAO9527/csf_project

cd csf_project

python3 setup.py install 
```

---
### Output:

    # 2 categories:
    # Misclassified samples: 84
    # The accuracy of random forest is: 0.46153846153846156

    # 3 categories:
    # Misclassified samples: 90
    # The accuracy of random forest is: 0.4230769230769231
