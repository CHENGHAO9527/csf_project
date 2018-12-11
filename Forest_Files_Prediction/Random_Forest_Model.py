from sklearn.ensemble import RandomForestClassifier

### 2 categories
# Declare the classifier 
rf2 = RandomForestClassifier(random_state=0, class_weight='balanced')

# Fit the model
rf2.fit(X2_train,y2_train)

y2_pred_rf = rf2.predict(X2_test)
print('Two categories:')
print('Misclassified samples: %d' % (y2_test != y2_pred_rf).sum())

# Print the accuracy
print('The accuracy of random forest is: ' + 
      str(rf2.score(X2_test, y2_test)))

# Plot feature importances
importances2 = rf2.feature_importances_
plot_feature_importances(features, importances2)

### 3 categories
# Declare the classifier 
rf3 = RandomForestClassifier(random_state=0, class_weight='balanced')

# Fit the model
rf3.fit(X3_train,y3_train)

y3_pred_rf = rf3.predict(X3_test)
print('')
print('Three categories:')
print('Misclassified samples: %d' % (y3_test != y3_pred_rf).sum())

# Print the accuracy
print('The accuracy of random forest is: ' + 
      str(rf3.score(X3_test, y3_test)))

# Plot feature importances
importances3 = rf3.feature_importances_
plot_feature_importances(features, importances3)

# 2 categories:
# Misclassified samples: 61
# The accuracy of random forest is: 0.6089743589743589

# 3 categories:
# Misclassified samples: 70
# The accuracy of random forest is: 0.5512820512820513
