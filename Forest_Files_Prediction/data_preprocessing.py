import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

# Load the data
def ffp_csf():
    original_df = pd.read_csv('http://www.dsi.uminho.pt/~pcortez/forestfires/forestfires.csv')

    # Show the header and the first five rows
    original_df.head()

    # rename 'X' and 'Y' to avoid confusion
    original_df.rename(index=str, columns={"X": "x_coord", "Y": "y_coord"},
    inplace=True)


    print('Number of rows before removing rows with missing values: ' +
    str(original_df.shape[0]))

    # Replace ? with np.NaN
    original_df.replace('?', np.NaN, inplace=True)

    # Remove rows with np.NaN
    original_df.dropna(how='any', inplace=True)

    print('Number of rows after removing rows with missing values: ' +
    str(original_df.shape[0]))

    # first categorize area
    def categorizeArea2Cats(area):
        if area == 0:
            return 'no_fire'
        else:
            return 'fire'

    def categorizeArea3Cats(area):
        if area == 0:
            return 'no_fire'
        elif (area > 0) & (area < 100):
            return 'small_fire'
        else:
            return 'large_fire'
    # Make two dataframes - one with 2 categories and one with 3
    df2cats = original_df.copy() # make a copy to keep things neat
    df2cats['area'] = df2cats['area'].apply(categorizeArea2Cats)

    # print out fire/no_fire counts
    categoryCount2 = df2cats.groupby('area').area.count()
    print('Two Categories: ')
    print(categoryCount2.head())

    df3cats = original_df.copy() # make a copy to keep things neat
    df3cats['area'] = df3cats['area'].apply(categorizeArea3Cats)

    # print out fire/no_fire counts
    categoryCount3 = df3cats.groupby('area').area.count()
    print('')
    print('Three Categories: ')
    print(categoryCount3.head())
    return True

