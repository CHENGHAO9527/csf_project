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

# We found this code in the course's github page, here:
# https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/code/ch03/ch03.ipynb

from matplotlib.colors import ListedColormap


def get_label(class_num):
    # as we recall from section 4 above:
    # Unencoded class labels: ['large_fire' 'no_fire' 'small_fire']
    # Encoded class labels: [0 1 2]
    if class_num == 0:
        return 'large fire'
    elif class_num == 1:
        return 'no fire'
    else:
        return 'small fire'


def plot_decision_regions(X_in, y_in, classifier, test_idx=None,
                          resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'orange', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y_in))])

    # plot the decision surface
    x1_min, x1_max = X_in[:, 0].min() - 1, X_in[:, 0].max() + 1
    x2_min, x2_max = X_in[:, 1].min() - 1, X_in[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y_in)):
        plt.scatter(x=X_in[y_in == cl, 0],
                    y=X_in[y_in == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=get_label(cl),
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X_in[test_idx, :], y_in[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


# visualize the results
# First use PCA to reduce dimensionality from 8 to 2 so we can plot
pca = PCA(n_components=2)

### 2 categories
X2_train_pca = pca.fit_transform(X2_train)
X2_test_pca = pca.transform(X2_test)

# Declare the classifier (we have to make a new one since 'rf'
# expects 8 features and plot_decision_regions expects 2)
rf2_pca = RandomForestClassifier(random_state=0, class_weight='balanced')
rf2_pca.fit(X2_train_pca, y2_train)

y2_pred_rf2_pca = rf2_pca.predict(X2_test_pca)
print('Two categories:')
print('Misclassified samples: %d' % (y2_test != y2_pred_rf2_pca).sum())

# Print the accuracy
print('The accuracy of random forest is: ' +
      str(rf2_pca.score(X2_test_pca, y2_test)))

X2_combined = np.vstack((X2_train_pca, X2_test_pca))
y2_combined = np.hstack((y2_train, y2_test))
test_start2 = X2_train_pca.shape[0]
test_end2 = X2_train_pca.shape[0] + X2_test_pca.shape[0]

print(X2_combined.shape)
plot_decision_regions(X_in=X2_combined, y_in=y2_combined,
                      classifier=rf2_pca,
                      test_idx=range(test_start2, test_end2))
### 3 categories
X3_train_pca = pca.fit_transform(X3_train)
X3_test_pca = pca.transform(X3_test)

# Declare the classifier (we have to make a new one since 'rf'
# expects 8 features and plot_decision_regions expects 2)
rf3_pca = RandomForestClassifier(random_state=0, class_weight='balanced')
rf3_pca.fit(X3_train_pca, y3_train)

y3_pred_rf3_pca = rf3_pca.predict(X3_test_pca)
print('Three categories:')
print('Misclassified samples: %d' % (y3_test != y3_pred_rf3_pca).sum())

# Print the accuracy
print('The accuracy of random forest is: ' +
      str(rf3_pca.score(X3_test_pca, y3_test)))

X3_combined = np.vstack((X3_train_pca, X3_test_pca))
y3_combined = np.hstack((y3_train, y3_test))
test_start3 = X3_train_pca.shape[0]
test_end3 = X3_train_pca.shape[0] + X3_test_pca.shape[0]

print(X3_combined.shape)
plot_decision_regions(X_in=X3_combined, y_in=y3_combined,
                      classifier=rf3_pca,
                      test_idx=range(test_start3, test_end3))

# 2 categories:
# Misclassified samples: 84
# The accuracy of random forest is: 0.46153846153846156

# 3 categories:
# Misclassified samples: 90
# The accuracy of random forest is: 0.4230769230769231