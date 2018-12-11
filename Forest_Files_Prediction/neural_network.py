from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support

### Two categories
# Declare the classifier 
mlp2 = MLPClassifier(random_state=0, hidden_layer_sizes=200, 
                    activation='relu')

# Fit the model
mlp2.fit(X2_train,y2_train)
y2_pred_mlp = mlp2.predict(X2_test)
print('Two categories:')
print('Misclassified samples: %d' % (y2_test != y2_pred_mlp).sum())

# Print the accuracy
print('The accuracy of neural net:', end=' ')
print(precision_recall_fscore_support(y2_pred_mlp, y2_test, 
                                      average='micro')[0])

### Three categories
# Declare the classifier 
mlp3 = MLPClassifier(random_state=0, hidden_layer_sizes=200, 
                    activation='relu')

# Fit the model
mlp3.fit(X3_train,y3_train)
y3_pred_mlp = mlp3.predict(X3_test)
print('')
print('Three categories:')
print('Misclassified samples: %d' % (y3_test != y3_pred_mlp).sum())

# Print the accuracy
print('The accuracy of neural net:', end=' ')
print(precision_recall_fscore_support(y3_pred_mlp, y3_test, 
                                      average='micro')[0])
