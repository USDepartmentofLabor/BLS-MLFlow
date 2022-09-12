# Standard Modules 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore", category = UserWarning)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.pyfunc
import mlflow.sklearn

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

os.environ["MLFLOW_TRACKING_URI"]="http://<Remote IP>:<Port>"
os.environ["MLFLOW_EXPERIMENT_NAME"]="sklearn-onet-experiment"
experiment = mlflow.get_experiment_by_name("sklearn-onet-experiment")
client = mlflow.tracking.MlflowClient()
run = client.create_run(experiment.experiment_id)

with mlflow.start_run(run_id = run.info.run_id):

    def preprocess(onet_df):
        stopwords_list = stopwords.words('english')
        onet_df['Task'] = onet_df['Task'].str.replace(r'[^\w\s]+', '')
        onet_df['Task'] = onet_df['Task'].apply(lambda x: ' '.join([word for word in x.split() if word not in 
                                                                    (stopwords_list)]))
        onet_df = pd.get_dummies(onet_df, columns=['GWA'], prefix='', prefix_sep='')
        onet_df = onet_df.groupby(by='Task').sum()
        onet_df[onet_df >= 1] = 1
        onet_df.reset_index(level=0, inplace=True)
        onet_df = onet_df.drop_duplicates()
    
        return onet_df

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
            if len(set_true) == 0 and len(set_pred) == 0:
                tmp_a = 1
            else:
                tmp_a = len(set_true.intersection(set_pred)) / len(set_true.union(set_pred))
            acc_list.append(tmp_a)

        return round(np.mean(acc_list),5)

    def prf_to_dict(prf_results, average='micro'):
        if average == 'sample':
            keys = ["Precision_sample", "Recall_sample", "F1_sample", "Support_sample"]
        else:
            keys = ["Precision_micro", "Recall_micro", "F1_micro", "Support_micro"]

        values = list(prf_results)
        metrics_dict = {}

        for key, value in zip(keys, values):
            metrics_dict[key] = value
            if value == None:
                metrics_dict[key] = -99.99
        return metrics_dict

    if __name__ == "__main__":
        print("Tracking URI: ", mlflow.get_tracking_uri())    
        onet_df = pd.read_parquet("../data/onet_task_gwa.pqt")
        onet_df = onet_df[['Task', 'GWA']]
        onet_df = preprocess(onet_df)
        mlflow.log_param("Dataset Shape", onet_df.shape)
        
        task_train, task_test, gwa_train, gwa_test = train_test_split(onet_df['Task'],
                                                                      onet_df[onet_df.columns.difference(['Task'])], 
                                                                      test_size = 0.20, random_state = 607, shuffle=True)
        
        task_train, sig_input, gwa_train, sig_pred = train_test_split(task_train, gwa_train, test_size = 0.01, random_state =
                                                                      607, shuffle=True)
        gwa_test.reset_index(level=0, drop=True, inplace=True)

        mlflow.log_param("Training Data Size", gwa_train.shape[0])
        mlflow.log_param("Testing Data Size", gwa_test.shape[0])
    

        C_vals = str(sys.argv[1]) if len(sys.argv) > 1 else "1, 10, 100"
        penalty_vals = str(sys.argv[2]) if len(sys.argv) > 2 else "none, l2"
        C_vals = list(int(x) for x in C_vals.split(", "))
        penalty_vals = list(x for x in penalty_vals.split(", "))
        grid_params = {"cls__C": C_vals, "cls__penalty": penalty_vals}
        predicted_prob = {}
        best_cv_params = {}
        
        pipeline = Pipeline([('tf',TfidfVectorizer(lowercase=True)),
                             ('cls', LogisticRegression(class_weight='balanced',solver='lbfgs', 
                                                        max_iter=300,random_state=607))])
        
        cross_val = GridSearchCV(pipeline, grid_params, scoring='f1_micro', cv=3)
        model_parameters = pipeline[1].get_params()
        mlflow.log_params(model_parameters)
        
        pipeline.fit(task_train, gwa_train['4A4c3'])
        sample_preds = pipeline.predict(task_test)
        
        sig_input = pd.DataFrame(sig_input)
        signature = mlflow.models.infer_signature(sig_input, sample_preds)
        mlflow.sklearn.log_model(pipeline, "logreg_model", conda_env='conda.yaml', signature=signature)

        for gwa in gwa_train.columns:
            cross_val.fit(task_train, gwa_train[gwa])

            gwa_parameters = cross_val.best_params_
            best_cv_params[gwa] = gwa_parameters 

            final_model = cross_val.best_estimator_
            probabilities = [x[1] for x in final_model.predict_proba(task_test)]
            predicted_prob[gwa] = probabilities 

        mlflow.log_params(best_cv_params)
        predicted_prob_df = force_prediction(pd.DataFrame.from_dict(predicted_prob))

        prf_micro = prf_to_dict(metrics.precision_recall_fscore_support(predicted_prob_df, gwa_test, 
                                                                                average = 'micro'), 'micro')
        prf_sample = prf_to_dict(metrics.precision_recall_fscore_support(predicted_prob_df, gwa_test, 
                                                                                  average = 'samples'), 'sample')

        accuracy_score = metrics.accuracy_score(predicted_prob_df, gwa_test)
        hamming_score_result = hamming_score(predicted_prob_df, gwa_test)
        hamming_loss = metrics.hamming_loss(predicted_prob_df, gwa_test)

        mlflow.log_metric("Accuracy", accuracy_score)
        mlflow.log_metrics(prf_micro)
        mlflow.log_metrics(prf_sample)
        mlflow.log_metric("Hamming Score", hamming_score_result)
        mlflow.log_metric("Hamming Loss", hamming_loss)

        row_sums = predicted_prob_df.sum(axis=1)
        row_sum_plot = row_sums.value_counts().plot(kind='bar')
        row_sum_plot = row_sum_plot.get_figure()
        mlflow.log_figure(row_sum_plot, "row_sum_plot.png")
        mlflow.log_artifact("mlproject_onet.py")

        mlflow.end_run()
