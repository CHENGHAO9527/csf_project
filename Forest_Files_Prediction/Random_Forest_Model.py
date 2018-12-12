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

