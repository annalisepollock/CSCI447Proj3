import math
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class Cleaner:
    def __init__(self):
        self.name = "Cleaner"

    def clean(self, dataFrame, dropColumns, classCol):
        # ADDRESS NULL VALUES WHERE COLUMNS/ROWS NEED TO BE REMOVED
        # If true class is unknown, drop the row
        cleanedData = dataFrame.dropna(subset=[classCol])
        subsetColumns = [col for col in cleanedData.columns if col != classCol]
        # Columns must have 70% of their values for rows to remain in dataset
        cleanedData = cleanedData.dropna(axis=1, thresh = math.floor(0.70*cleanedData.shape[0]))

        # Drop any rows where all values are null
        cleanedData = cleanedData.dropna(how = 'all', subset=subsetColumns)

        # Remove unnecessary columns (i.e., ID columns)
        if len(dropColumns) > 0:
            cleanedData = cleanedData.drop(columns=dropColumns, axis = 1)

        # Get list of categorical column names
        classColDataFrame = cleanedData[classCol]
        cleanedData = cleanedData.drop(columns=classCol, axis = 1)
        categoricalColumns = cleanedData.select_dtypes(exclude=['int', 'float']).columns.tolist()

        # One Hot Encoding
        # cleanedData = pd.get_dummies(cleanedData, columns=categoricalColumns, dtype=int)
        encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded = encoder.fit_transform(cleanedData[categoricalColumns])
        encodedData = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categoricalColumns))
        cleanedData = pd.concat([cleanedData, encodedData], axis=1)
        cleanedData = cleanedData.drop(categoricalColumns, axis=1)

        # Iterate through columns and fill NaN values for string/object columns
        for col in cleanedData.select_dtypes(include='object'):
            modeVal = cleanedData[col].mode()[0]  # Get the mode (most frequent value)
            cleanedData[col] = cleanedData[col].fillna(modeVal)  # Fill NaN with the mode

        # Fill all other na values with the mean of each column
        cleanedData = cleanedData.fillna(cleanedData.mean())
        cleanedData[classCol] = classColDataFrame

        # Z-score normalization
        normalizeColumns = list(cleanedData.columns)
        normalizeColumns.remove(classCol)
        one_hot_encoded_columns = list(encodedData.columns)
        normalizeColumns = [col for col in normalizeColumns if col not in one_hot_encoded_columns]

        for col in normalizeColumns:
            col_zscore = col + '_zscore'
            
            # set zscore to zero if standard deviation is zero, calculate normally if not zero
            if ((cleanedData[col].std(ddof=0)) != 0):
                cleanedData[col_zscore] = (cleanedData[col] - cleanedData[col].mean())/cleanedData[col].std(ddof=0)
            else:
                cleanedData[col_zscore] = 0
                
        
        cleanedData = cleanedData.drop(columns=normalizeColumns, axis=1)

        return cleanedData