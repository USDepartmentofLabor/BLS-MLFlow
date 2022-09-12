# BLS MLflow Projects & Model Deployment Documentation 
*Remy Stewart, BLS Civic Digital Fellow, Summer 2022*

This subdirectory of the bls-mlflow pilot integration repository demonstrates how to use the MLflow Projects feature to run a MLflow model on the command line as well as how to serve the model as a local REST API. 

## 1.0 MLflow Projects
MLflow Projects are directories configured following MLflow's specified project file and code format that allow the platform to run the directory and reproduce machine learning model scripts. They consist of three primary files as demonstrated within this folder- a MLproject text file unique to MLflow, a Python file with the modeling code, and a environment configuration file such as a conda or python_env file. 

The MLproject file instructs MLflow towards how to run the model code specified through a Python file such as mlproject_onet.py. Let's review the structure of the file itself:
```
name: mlproject_onet

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      C: {type: string, default: "1, 10, 100"}
      penalty: {type: string, default: "none, l2"}
    command: "python mlproject_onet.py {C} {penalty}"    
```
We specify the name of the project and this folder's conda.yaml file to create our virtual environment for our modeling script.   

**Entry points** delineate how the model code should be run when the MLflow Project is called and whether it includes input parameters into the model that can be configured across project runs. The model featured in mlproject_onet.py is a logistic regression model that performs a small hyperparameter search on whether or not a regularization penalty is applied, as well as the strength of implemented penalties as set by the C term. We therefore set two parameters within the MLproject file to allow changes to these hyperparameter search spaces across runs and include them as arguments in the specified command syntax for our MLflow Project. 

Entry point parameters are established within MLproject files via dictionaries. The first key-value pair specifies the expected value type for the parameter, with MLflow supporting strings, floats, file paths, and HTTP URLs. The second pair provides default values for the parameter if a user running a project does not directly set the values themselves. 

mlproject_onet.py is an adapted version of the code featured in the first Scikit-learn demo notebook within this repository's examples folder. It primarily consists of the same code with a few modifications to support its use as a MLflow Project:
- Running Projects folders automatically starts a MLflow model run, so the first code chunk following importing the requisite libraries is designed to link the automatically launched local run to the remote BLS server. We first set the server's URL and the experiment we'd like to log our run under as environmental variables. We then create a MLflow Client object to manually link our local run to the experiment directory within the remote server.
 - The functions from the helpers.py file within the examples folder have been directly added into the script to ensure they're consistently accessible to the main modeling code within different production environments. 
 - We draw from Scikit-learn's [Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline) feature to combine the TF-IDF vectorizer we use to convert the occupation associated task text from the public ONET data with our one-vs.-all logistic regression classifier.  This guarantees that the vectorizer used to train the model will always be loaded in with the model itself when our project is deployed. 
 - There's a few additional edits within the script to support running the model as a MLflow Project, such as converting the entry point parameter inputs into dictionary values for the hyperparameter grid search.

## 2.0 Running Projects 

Now that we've reviewed the necessary file structure to create an MLflow Project, let's test out launching the project itself within a terminal window. MLProjects can be called from a local file path or directly from Git repositories, although local files will likely be better suited for BLS use cases. For the local file approach following a `git clone` of this repository, the command is `mlflow run './mlproject'` after CD'ing into the bls-mlflow folder directly before this subdirectory. 

The project will begin by creating its conda virtual environment and then move to running the modeling script. It'll take a few minutes to go through the model training loop that includes the hyperparameter search in which no output will be printed within the terminal. You'll know that the project was successfully ran after a log message similar to below is printed:
```bash
2022/07/19 09:19:00 INFO mlflow.projects: === Run (ID '<Run ID>') succeeded ===
```
The model run will be referable directly within the MLflow Tracking UI at the remote server's URI of `http://<Remote IP>:<Port>` as well. 

## 3.0 Model Deployment

Now that we've successfully ran our MLflow Project and logged the model within MLflow, we can move to locally deploying our model. This is facilitated by the `mlflow models serve` command: 
```bash
mlflow models serve --model-uri <Artifact Path> --port <####>
```
The call specifies the artifact path on the remote server that houses our logged model, which can be found through this run's associated page within the MLflow UI. We additionally set the local port for the deployed model to send data to to test the model's ability to generate real-time predictions. The following output confirms that a Gunicorn local server was successfully launched that now hosts our model: 
```bash
<Local IP>:<Port> -w 1 ${GUNICORN_CMD_ARGS} -- mlflow.pyfunc.scoring_server.wsgi:app'
[2022-07-19 11:07:13 -0400] [2669324] [INFO] Starting gunicorn 20.1.0
[2022-07-19 11:07:13 -0400] [2669324] [INFO] Listening at: http://<Local IP>:<Port> (2669324)
```
Let's send a test data sample to our deployed model to verify whether it can produce robust predictions on unseen data. Since we've nested both our text vectorizer and our logistic classifier into one live pipeline, we can send over a raw string sample through a cURL call. Following the Pandas DataFrame format the model was trained on, we specify the column of our task text and create a previously unseen task description for the model. 
```bash
curl -d "{\"columns\":[\"Task\"],\"index\":[0],\"data\":[\"Build and deploy machine learning models for the Bureau of Labor Statistics.\"]}" -H "Content-Type: application/json" http://<Local IP>:<Port>/invocations

```
If you see `[0]` following this cURL call, that means the model has successfully generated a prediction on the test string. Within the Scikit-learn example notebook this code is sourced from, the logged model is the last model out of the 37 models cycled through the one-vs.-all prediction task. We're therefore receiving a single prediction for class membership regarding this General Work Activity from the ONET data.

