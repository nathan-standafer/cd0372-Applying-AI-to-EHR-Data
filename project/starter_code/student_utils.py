import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''

    ndc_code_df = ndc_df[['NDC_Code', 'Non-proprietary Name']]
    ndc_code_df = ndc_code_df.rename(columns={'NDC_Code':'ndc_code'})
    
    reduce_dim_df = pd.merge(df, ndc_code_df[['ndc_code', 'Non-proprietary Name']], on='ndc_code', how='left')
    reduce_dim_df = reduce_dim_df.rename(columns={'Non-proprietary Name':'generic_drug_name'})
    
    return reduce_dim_df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    reduce_dim_df_sorted = df.sort_values(by=['patient_nbr', 'encounter_id'])
    #reduce_dim_df_sorted.head(50)
    
    first_occurrence_rows = reduce_dim_df_sorted.groupby('patient_nbr').apply(lambda x: x.iloc[0])
    first_occurrence_rows = first_occurrence_rows.reset_index(drop=True)

    return first_occurrence_rows


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    # First split off 40% of the dataset for testing and validating
    train, df_temp = train_test_split(df, test_size=0.4, random_state=1)
    
    # Now further divide the remaining (60%) into test and validation sets
    test, validation = train_test_split(df_temp, test_size=0.5, random_state=1)

    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
                                                                                    c,
                                                                                    vocab_file_path)

        
        output_tf_list.append(tf.feature_column.indicator_column(tf_categorical_feature_column))
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean=0, std=0):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std

from functools import partial
def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalize_numeric_with_zscore_partial = partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(col, default_value=default_value, normalizer_fn=normalize_numeric_with_zscore_partial)
    
    return tf_numeric_feature

   
#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''

    student_binary_prediction = pd.DataFrame([[int(i > 5)] for i in df[col]], columns=['binary_pred'])
    
    return student_binary_prediction
