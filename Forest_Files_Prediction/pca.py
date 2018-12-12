# visualize the results
# Remeber to use the dimensionality-reduced PCA versions of X

### 2 categories
# Declare the classifier (we have to make a new one since 'mlp' 
# expects 8 features and plot_decision_regions expects 2)
mlp2_pca = MLPClassifier(random_state=0, hidden_layer_sizes=200, 
                    activation='relu')
# Fit the model
mlp2_pca.fit(X2_train_pca, y2_train)

y2_pred_mlp2_pca = mlp2_pca.predict(X2_test_pca)
print('Two categories')
print('Misclassified samples: %d' % (y2_test != y2_pred_mlp2_pca).sum())

# Print the accuracy
print('The accuracy of mlp2_pca is: ' + 
      str(mlp2_pca.score(X2_test_pca, y2_test)))
    
plot_decision_regions(X_in=X2_combined, y_in=y2_combined, 
                      classifier=mlp2_pca, 
                      test_idx=range(test_start2, test_end2))

### 3 categories
# Declare the classifier (we have to make a new one since 'mlp' 
# expects 8 features and plot_decision_regions expects 2)
mlp3_pca = MLPClassifier(random_state=0, hidden_layer_sizes=200, 
                    activation='relu')
# Fit the model
mlp3_pca.fit(X3_train_pca, y3_train)

y3_pred_mlp3_pca = mlp3_pca.predict(X3_test_pca)
print('Three categories')
print('Misclassified samples: %d' % (y3_test != y3_pred_mlp3_pca).sum())

# Print the accuracy
print('The accuracy of mlp3_pca is: ' + 
      str(mlp3_pca.score(X3_test_pca, y3_test)))
    
plot_decision_regions(X_in=X3_combined, y_in=y3_combined, 
                      classifier=mlp3_pca, 
                      test_idx=range(test_start3, test_end3))

# 2 categories:
# Misclassified samples: 71
# The accuracy of mlp2_pca is: 0.5448717948717948

# 3 categories:
# Misclassified samples: 110
# The accuracy of mlp3_pca is: 0.2948717948717949
