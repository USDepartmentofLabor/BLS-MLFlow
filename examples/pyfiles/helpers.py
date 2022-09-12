# Standard modules
import pandas as pd
import matplotlib.pyplot as plt
import collections
import numpy as np
import pickle

# Text Pre-processing modules
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 

# Hugging Face Datasets
from datasets import Dataset

# Mlflow
import mlflow

def preprocess(onet_df):
    # NLTK English stopwords object
    stopwords_list = stopwords.words('english')

    # Removing punctation & stopwords
    onet_df['Task'] = onet_df['Task'].str.replace(r'[^\w\s]+', '')
    onet_df['Task'] = onet_df['Task'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_list)]))

    # Aggregating GWA labels by task 
    onet_df = pd.get_dummies(onet_df, columns=['GWA'], prefix='', prefix_sep='')
    onet_df = onet_df.groupby(by='Task').sum()
    onet_df[onet_df >= 1] = 1
    onet_df.reset_index(level=0, inplace=True)
    onet_df = onet_df.drop_duplicates()
    
    return onet_df

def split_and_encode(df, text_col):
    # Training & testing sets for model development
    task_train, task_test, gwa_train, gwa_test = train_test_split(df[text_col],
                                                                  df[df.columns.difference([text_col])], test_size =
                                                                  0.20, random_state = 607, shuffle=True)
    # Small prediction set
    task_train, task_pred, gwa_train, gwa_pred = train_test_split(task_train, gwa_train, test_size = 0.02, random_state = 607, 
                                                                  shuffle=True)
    # Vectorize train, test, and prediction tasks separately
    text_vectorizer = TfidfVectorizer(lowercase=True)

    task_train = text_vectorizer.fit_transform(task_train)
    task_test = text_vectorizer.transform(task_test)

    # Reset index for metric calculations
    gwa_test.reset_index(level=0, drop=True, inplace=True)
    
    return task_train, task_test, gwa_train, gwa_test, task_pred, text_vectorizer

def force_prediction(predicted_probs_df):
    '''
    Function to ensure that all tasks are assigned to at least one GWA within the One vs. All models. Converts all predicted 
    probabilities > 0.5 to 1 and takes the largest probability in instances where all predictions are <= 0.5.
    '''
    predicted_probs_df[predicted_probs_df > 0.5] = 1
    predicted_prob_df_nomax = predicted_probs_df[predicted_probs_df.max(axis=1) < 0.5]
    predicted_prob_df_nomax.values[range(len(predicted_prob_df_nomax.index)), 
                                          np.argmax(predicted_prob_df_nomax.values, axis=1)] = 1
    all_predicted = predicted_prob_df_nomax.combine_first(predicted_probs_df)
    all_predicted[all_predicted != 1] = 0
    return all_predicted

def hamming_score(true_df, pred_df): 
    '''
    Function code is sourced from StackOverflow: https://stackoverflow.com/q/3223957/
    '''
    acc_list = []
    for i in range(true_df.to_numpy().shape[0]):
        set_true = set(np.where(true_df.to_numpy()[i])[0])
        set_pred = set(np.where(pred_df.to_numpy()[i])[0])

        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0: # conditional to account for the division by zero error
            tmp_a = 1 # in practice, this condition should not occur since set_true should contain at least one gwa
        else:
            tmp_a = len(set_true.intersection(set_pred)) / len(set_true.union(set_pred))
        acc_list.append(tmp_a)

    return round(np.mean(acc_list),5)

def prf_to_dict(prf_results, average='micro'):
    '''
    MLflow's log_metrics requires a dictionary input. This function maps the four output values of sklearn's 
    precision_recall_fscore_support to a dictionary format for both the micro & sample average.
    '''
    if average == 'sample':
        keys = ["Precision_sample", "Recall_sample", "F1_sample", "Support_sample"]
    elif average=='macro':
        keys = ["Precision_macro", "Recall_macro", "F1_macro", "Support_macro"]
    else:
        keys = ["Precision_micro", "Recall_micro", "F1_micro", "Support_micro"]
        
    values = list(prf_results)
    dictionary = {}

    for key, value in zip(keys, values):
        dictionary[key] = value
        if value == None:
            # The support metric doesn't apply to multilabel classification and I therefore set the value to a NA placeholder
            # that MLflow accepts.
            dictionary[key] = -99.99
    return dictionary

def hf_data_processing(onet_base): 
    '''
    Preprocessing workflow for the Hugging Face transformers example notebook on the ONET dataset. Handles separating the 
    back-translated task text to avoid data leakage and converting Pandas data frames into transformer-compatable Datasets. 
    '''
    # generating one-hot encoding of GWAs and recombining with backtranslated tasks
    onet_df = pd.get_dummies(onet_base, columns=['GWA'], prefix='', prefix_sep='')
    onet_df = onet_df.groupby(by='Task', as_index=False).sum()
    onet_task_unique = onet_base.drop_duplicates(['Task'])[['Task', 'Task_backtranslated']]
    onet_df = onet_df.merge(onet_task_unique, how='left', on='Task')
    
    # GWA label value list & column conversion to integers
    labels = [column for column in onet_df if column.startswith('4A')]
    
    for gwa in labels:
        onet_df[gwa] = onet_df[gwa].astype(np.int32)
    
    # Training/testing/prediction split, removing backtranslated values to prevent data leakage
    training, prediction = train_test_split(onet_df, test_size=0.02, shuffle=True, random_state=607)
    prediction.drop(['Task_backtranslated'], axis=1, inplace=True)
    prediction.reset_index(drop=True, inplace=True)
    training, testing = train_test_split(training, test_size=0.20, shuffle=True, random_state=607)
    testing.drop(['Task_backtranslated'], axis=1, inplace=True)
    testing.reset_index(drop=True, inplace=True)
    
    # Reconcatenating backtranslated text to training set
    training_bt = training[['Task_backtranslated']+labels]
    training_bt = training_bt.rename(columns = {'Task_backtranslated':'Task'})
    training_bt = training_bt.drop_duplicates(['Task'])
    training = pd.concat([training, training_bt])
    training.drop(['Task_backtranslated'], axis=1, inplace=True)
    training.reset_index(drop=True, inplace=True)
    
    # Converting Training & Testing to Huggingface Datasets
    training = Dataset.from_pandas(training) 
    testing = Dataset.from_pandas(testing)
    
    print("Training Size:", len(training))
    print("Testing Size:", len(testing))
    print("Prediction Size:", len(prediction))
    
    return training, testing, prediction, labels
