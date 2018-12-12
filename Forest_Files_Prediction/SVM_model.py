from sklearn.svm import SVC
From Forest_Files_Prediction.data_preprocessing import ffp_csf
From Forest_Files_Prediction.get_feature import ffp_get_feature,ffp_train_test_split\

### Two categories
# Declare the classifier 
def ffp_svm_model():
    svm2 = SVC(class_weight='balanced', random_state=0, probability=True)

    # Fit the model
    svm2.fit(X2_train, y2_train)

    y2_pred_svm = svm2.predict(X2_test)
    print('Two categories:')
    print('Misclassified samples: %d' % (y2_test != y2_pred_svm).sum())

    # Print the accuracy
    print('The accuracy of SVM is: ' + str(svm2.score(X2_test, y2_test)))


    ### Two categories
    # Declare the classifier
    svm3 = SVC(class_weight='balanced', random_state=0, probability=True)

    # Fit the model
    svm3.fit(X3_train, y3_train)

    y3_pred_svm = svm3.predict(X3_test)
    print('')
    print('Three categories:')
    print('Misclassified samples: %d' % (y3_test != y3_pred_svm).sum())

    # Print the accuracy
    print('The accuracy of SVM is: ' + str(svm3.score(X3_test, y3_test)))

    # 2 categories:
    # Misclassified samples: 69
    # The accuracy of SVM is: 0.5576923076923077

    # 3 categories:
    # Misclassified samples: 90
    # The accuracy of SVM is: 0.4230769230769231
    return True
