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

    def plot_2d_space(X, y, label='Classes', num_cats=2):
        pca = PCA(n_components=2)
        twoDX = pca.fit_transform(X)
        cat_indexes = []
        cat_indexes.append([index for index in range(len(y)) if y[index] == 1])
        if num_cats == 3:
            cat_indexes.append([index for index in range(len(y)) if y[index] == 2])
        cat_indexes.append([index for index in range(len(y)) if y[index] == 0])

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
