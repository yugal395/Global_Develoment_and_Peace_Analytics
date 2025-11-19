import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
 # "load the dataset"
    df=pd.read_csv(file_path)
    print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_structure(df):
    "Fix column names and remove duplicates"
    df.columns=(
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace('%', 'percent')
    )
    df.drop_duplicates(inplace=True) #removes repeated indentical rows. built in function
    print("Column names cleaned and duplicates removed.")
    return df
def handle_missing_values(df):
    #"handle missing values
    threshold=len(df)*0.7  #calculates numbber of rows(records) in the dataset and multiplies by 0.7 , meaning 70% of the total rows, main idea is to set a threshold that maybe if 70% rows are missing we remove that column
    df=df.dropna(axis=1,thresh=threshold)
    for col in df.select_dtypes(include=[np.number]).columns: #selects only numerical columns
        df[col].fillna(df[col].median(),inplace=True) #fills missing values with median
        print(f"Missing values in numerical column '{col}' filled with median.")
        return df
    
def fix_data_types(df):
    #"correct data types"
  if 'date' in df.columns:
    df['date']=pd.to_datetime(df['date'],errors='coerce').dt.year #extracts only the year part of it, #converts to datetime format, errors coerce means if there is an error it will set it to NaT
    print("Data types fixed.")  
    return df
  def scale_numeric(df):#scales numerical values between 0 and 1 except years
   scaler=MinMaxScaler()
   num_cols=df.select_dtypes(include=[np.number]).columns #selects the numerical columns only
   if 'date' in num_cols:
    num_cols=num_cols.drop('date') #drops date column from numerical columns
    df[num_cols]=scaler.fit_transform(df[num_cols]) #fits and transforms the numerical columns
   

    
         