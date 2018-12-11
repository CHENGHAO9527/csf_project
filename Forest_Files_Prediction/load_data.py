import warnings
warnings.filterwarnings("ignore")
import pandas as pd

# Load the data
original_df = pd.read_csv('http://www.dsi.uminho.pt/~pcortez/forestfires/forestfires.csv')

# Show the header and the first five rows
original_df.head()
