From Forest_Files_Prediction.data_preprocessing import ffp_csf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Specify the name of the target and features
def ffp_get_feature():
    target = 'area'
    features = ['temp','RH', 'wind', 'rain', 'FFMC','DMC','DC','ISI']

    ### 2 categories
    # Get the target vector
    y2 = df2cats[target].values
    print('Two Categories: ')
    print('Unencoded class labels:', np.unique(y2))

    # Get the feature vector
    X2 = df2cats[features].values


    # Declare the LabelEncoder
    class_le = LabelEncoder()

    # Encode the target
    y2 = class_le.fit_transform(y2)
    print('Encoded class labels:', np.unique(y2))

    ### 3 categories
    # Get the target vector
    y3 = df3cats[target].values
    print('')
    print('Three Categories: ')
    print('Unencoded class labels:', np.unique(y2))

    # Get the feature vector
    X3 = df3cats[features].values


    # Declare the LabelEncoder
    class_le = LabelEncoder()

    # Encode the target
    y3 = class_le.fit_transform(y3)

    print('Encoded class labels:', np.unique(y3))
    return True

def ffp_train_test_split():
    from sklearn.model_selection import train_test_split

    ### 2 categories
    # Randomly choose 30% of the data for testing (set randome_state as 0 and stratify as y)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2,
                                                            test_size=0.3,
                                                            random_state=0,
                                                            stratify=y2)
    # Standardize the features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X2_train = scaler.fit_transform(X2_train)
    X2_test = scaler.transform(X2_test)

    print('Two Categories: ')
    print('Label counts in y2:', np.bincount(y2))
    print('Label counts in y2_train:', np.bincount(y2_train))
    print('Label counts in y2_test:', np.bincount(y2_test))

    ### 3 categories
    # Randomly choose 30% of the data for testing
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3,
                                                            test_size=0.3,
                                                            random_state=0,

                                                            stratify=y3)

    # Standardize the features
    X3_train = scaler.fit_transform(X3_train)
    X3_test = scaler.transform(X3_test)

    print('')
    print('Three Categories: ')
    print('Label counts in y3:', np.bincount(y3))
    print('Label counts in y3_train:', np.bincount(y3_train))
    print('Label counts in y3_test:', np.bincount(y3_test))

    # Look at the data
    % matplotlib
    inline
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    def plot_2d_space(X, y, label='Classes', num_cats=2):
        # make the data 2-dimensional for plotting
        pca = PCA(n_components=2)
        twoDX = pca.fit_transform(X)
        # make an array to hold the arrays of indexes for each category
        cat_indexes = []
        cat_indexes.append([index for index in range(len(y)) if y[index] == 1])
        if num_cats == 3:
            cat_indexes.append([index for index in range(len(y)) if y[index] == 2])
        cat_indexes.append([index for index in range(len(y)) if y[index] == 0])

        if num_cats == 2:
            colors = ['#1F77B4', '#FF7F0E']
            markers = ['o', 's']
            labels = ['no fire', 'fire']
        else:
            colors = ['#1F77B4', '#FF7F0E', 'black']
            markers = ['o', 's', 'x']
            labels = ['no fire', 'small fire', 'large fire']

        # now plot each category
        for idx in range(len(cat_indexes)):
            curr_dots = twoDX[cat_indexes[idx]]
            plt.scatter(
                curr_dots[:, 0],
                curr_dots[:, 1],
                c=colors[idx], label=labels[idx], marker=markers[idx]
            )
        plt.legend(loc='upper left')
        plt.title(label).set_fontsize(16)
        plt.tight_layout()
        plt.show()

    plot_2d_space(X2_train, y2_train, 'Imbalanced dataset - 2 categories',
                  num_cats=2)

    plot_2d_space(X3_train, y3_train, 'Imbalanced dataset - 3 categories',
                  num_cats=3)

    return True