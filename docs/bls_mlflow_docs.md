# BLS MLFlow Pilot Integration Documentation
*Remy Stewart, BLS Civic Digital Fellow, Summer 2022*

With the increase of machine learning applications across BLS offices, there is a growing need to centralize siloed ML workflows into a designated location to promote model reproducibility, collaboration across teams, and assist with transitioning models from experimentation to production environments. MLflow is an open-source platform that aims to provide both a central location to facilitate the practices of machine learning operations within organizations- commonly shortened to MLOps- as well as offer multiple features each serving different stages of machine learning model lifecycles.

This documentation serves as an introduction to MLflow overall and towards its pilot application within BLS. It provides an overview of MLflow’s key features, how they may support specific needs for BLS data scientists, and tips for how to best incorporate the platform’s components into everyday data science workflows. Following reviewing this documentation, you’ll find five Jupyter Notebooks within this repository's examples folder that offer code walkthroughs of how to integrate MLflow directly into Python scripts for both Scikit-learn and Hugging Face transformer models processing three different public data sets. The mlprojects folder additionally demostrates the MLflow Projects feature and how to deploy packaged model scripts into production environments. 

[[_TOC_]]

## What is MLOps?

![mlops.png](./imgs/mlops.png)

**MLOps** refers to a set of best practices to strengthen the reliability, transparency, and robustness of ML pipelines. It builds from the aligned concept of Development Operations (DevOps) within software engineering.

Much of the recent growing interest and application of data science have focused on the earlier stages of machine learning model lifecycles regarding building high-performing models that are never translated within any other setting outside of a data scientist’s individual workspace. Attempts to expand the potential usability of models by sharing results and collaborating on experiments is often sidetracked by obstacles such as non-reproducible models and issues around storing and retrieving models within different virtual environments.

MLOps aims to address these challenges by identifying common goals and protocols to operationalize robust ML applications. It has inspired the significant growth of a range of ML platforms and packages intended to meet different needs related to implementing MLOps, in which MLflow is one of the primary leaders within this community.

## Example Use Cases for MLflow

The following examples are common situations within machine learning that are ideal candidates for MLflow support.

1. An individual data scientist is running the same ML model code but is testing different hyperparameters in hopes of achieving the best model performance.

	- MLflow can track experimentation around optimal hyperparameter settings, recording both the entire search space as well as the best identified parameters.

2.  A data scientist is comparing the same modeling goal and performance metrics across entirely different models and ML libraries, such as accuracy & F1-scores between a XGBoost tree classifier versus a Pytorch neural network.

	- MLflow supports a diverse range of ML modeling libraries and easily captures the same performance metrics across model architectures.

3. Multiple data scientists want to collaborate on the same modeling project while ensuring they don’t repeat any of the work already completed by their teammates.

	- MLflow provides a central location for colleagues to record their experiment results for easy reference between collaborators. It also actively records the model itself, its virtual environment, and the code required to replicate how the model was obtained.

4. A robust model structure has been identified and is now ready to be tested for production use.

	- MLflow’s configuration of logged models allows them to be loaded into a range of deployment settings.

5.  A deployed model needs to be retrained with fresh data.

	- MLflow can preserve the original model within its tracking server that is readily accessible for updates without disrupting the live model. It will also keep the original model separate from the retrained model to ensure version control.

These are a few situations where MLflow can be of particular use. There are likely many additional examples beyond these five where MLflow may aid data science work within the BLS that will becoming increasingly transparent with greater adoption of the MLflow platform.

# Overview of MLflow

![mlflow_modules.png](./imgs/mlflow_modules.png)

MLflow features four primary modules- Tracking, Projects, Models, and Registry. Let’s explore the main features of each module in turn. The descriptions for all four modules are linked to the official MLflow documentation for each respective section for additional reading as well. 

## MLflow Tracking

[MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html) encompasses both an API and the server user interface (UI) that BLS data scientists can reference as the central location for models logged into MLflow. The API includes a variety of methods to record different components of your ML pipeline. The Tracking UI organizes projects, displays recorded metadata, and facilitates cross-model comparisons.

### What can be recorded via MLflow?

**Experiments** are often equivalent to a specific project within a data science team. **Runs** are individual instances of running a MLflow-configured script that is assigned to a given experiment. Each time you run a script with any of the MLflow Tracking API methods, a new run will be created automatically if one is not already active. All runs are assigned a Git hash ID that can be used to find the metadata associated with the run. Experiment and run management are facilitated by the `mlflow.set_experiment`, `mlflow.start_run`, and `mlflow.end_run` methods. 

**Parameters** are the “inputs” of your ML pipeline. These can refer to model attributes such as individually set or default model parameters, class weights, or hyperparameter search space values. They can additionally capture data set metadata such as its shape or descriptive statistics. Parameters can store a range of data types such as strings, integers, tuples, or dictionaries. The associated Tracking API method for logging parameters within your pipeline is `log_parameter`.

**Metrics** refer to “outputs” relevant for measuring your model’s performance such as  accuracy, F1 scores, and the outputs of custom metric functions. Metric values are strictly numeric and are captured via `mlflow.log_metric`.

**Tags** are  label and string parameters created within your script. This is an optional feature that can be used to add short text notes to model runs through calling `mlflow.set_tag`.

**Artifacts** are more complex object types such as model files themselves, virtual environment configurations, Jupyter notebooks and Python scripts, and data visualization PNG/JPG files. They are first stored into directories and then logged into the MLflow tracking server. Artifact logging therefore references a file path in contrast to parameters and metrics which are objects within the script itself. Artifacts are retrieved via `mlflow.log_artifact`.

### MLflow Tracking UI

The Mlflow UI provides a central interface for experiments, model runs, and their affiliated logged values to be referenced and compared to each other. The UI is accessed through a web browser via combining an IP address and port number similar to Jupyter Notebooks. Importing and activating MLflow through any API call will by default assign the UI to the local machine's 5000 port equivalent to ` http://<Local IP>:5000`. To configure the UI port location directly, you can use `mlflow ui --port <####>` with a new port number in a command prompt that has a mlflow-configured conda environment active, and then subsequently shut down the UI through `^C` within the same prompt. 

Data scientists may want to start with a local MLflow UI instance for the earliest stages of their project, or when starting out with integrating MLflow into their code. During this stage, users are likely to have many test runs of a script that are not for model tracking purposes itself but rather for ensuring proper configuration with the MLflow API. The central BLS server can instead track model runs that are intended for transparent experimentation and reference across teams. I recommend using the `mlflow ui` call above to create a local MLflow server focused on developing robust MLflow-intergrated scripts within the initial stages of a project. When a data scientist is then ready to focus on actual model experiementation and sharing their results, they can easily configure their code to record runs to the remote server rather than a local Mlflow server by using the `mlflow.set_tracking_uri` method within their Python script. 

MLflow’s tracking system is divided into a backend store for model runs and their associated parameters and metrics, with a separate artifact store for these larger saved files. Backend stores can either be configured to a local file path with the automatically created `./mlruns` after running `mlflow ui` within a specific directory, or can point to a variety of SQL database servers. A SQL backend store is required for certain MLflow features such as the Model Registry. Artifact stores by default point to the `./mlruns` directory which within a local MLflow server will be automatically generated within the directory the command line was CD'd within when calling `mlflow ui`.

### Setting Up the BLS Remote Mlflow Server

The BLS MLflow server is running within a conda virtual environment titled `bls-mlflow`, which can be replicated from this repository’s associated conda.yaml file. After creating and activating this conda environment which includes MLflow as a dependency via:

```python
conda env create -f conda.yaml
conda activate bls-mlflow 
```

The server was then instantiated through the following command line call:

```python
mlflow server --backend-store-uri postgresql://postgresql_user:postgresql_passwd@localhost:<port>/mlflow --serve-artifacts --artifacts-destination <artifact path> --host <Local IP> --port <####> &
```

The `--backend-store-uri` flags a designated PostgresSQL database configured within the server while `--artifacts-destination` points to the directory path where all artifacts are remotely logged. Please refer to the postgresql_setup.md file within this folder for instructions delineating how the PostgresSQL database was created. The `--serve-artifacts` flag authorizes proxy access to the local artifact path that allows users to log artifacts to the remote server without having to log in directly. This ensures that artifacts will be properly associated with their respective runs within the remote server’s UI as well.

Pasting the following URL into your local browser will direct you to the actively running central server: `http://<Remote Server>:<Port>/`.

This approach can be easily replicated to different web server IP addesses and ports via changing the value of the `--host` and `--port` flag respectively. This can be used to create separate remote servers across BLS teams if needed within later integration stages of MLflow. 

## MLflow Projects
[MLflow Projects](https://www.mlflow.org/docs/latest/projects.html) are either local folders or Git repositories configured to support reproducing modeling scripts with the same environment and dependencies the model was originally trained in. Creating an MLflow Project with its associated code and storing it on the BLS Gitlab allows you to use the `mlflow.projects.run` command pointing to the Gitlab URL to run the associated scripts directly or alternatively run the file after cloning the repository to your local machine. MLflow Projects include an environment configuration file such as conda.yaml and python_env.yaml, a .py script with specified code to run known as the *entry point*, as well as expected configurable parameters for the model. These components can be passed directly into a `mlflow.projects.run` call and are commonly specified via a MLproject file. You can refer to an example MLflow Projects repository within the mlprojects folder.


## MLflow Models

[MLflow Models](https://www.mlflow.org/docs/latest/models.html) encompasses MLflow’s model logging, packaging, and deployment module that allows for models to be easily retrieved and reproduced across teams. MLflow Models are stored as artifact directories within model runs that includes a central MLmodel file that establishes how the model can be loaded and deployed as well as its associated environment configuration files.

There's a high amount of conceptual overlap between MLflow Projects and Models. The main differentiator between the two is that MLflow Projects refers to the system of how MLflow can run full model scripts configured for MLflow logging directly, while MLflow Models facilitates the consistent packaging of the model themselves that can be retrieved from MLflow servers.

### Model Flavors
Model flavors offer pre-established methods for streamlining the saving and loading of models created through various ML libraries. There are a wide range of flavors across libraries commonly used within the BLS supported by MLflow, with the first example notebook within this repository highlighting the `mlflow.sklearn` model flavor.

Mlflow’s baseline `mlflow.pyfunc` model flavor provides both a standard means to load in any logged MLflow model as well as support for custom model flavors. This is a key resource for saving and loading models that do not have pre-established model flavors within MLflow. The fourth example notebook within this repository demonstrates how to tailor`mlflow.pyfunc` for a Hugging Face transformers model, as transformer models is not a currently supported model flavor within MLflow.

### Autologging
Multiple of MLflow model flavors include an autologging feature that automates the capturing of parameters, metrics, and artifacts within supported ML libraries such as Scikit-learn and Pytorch. This feature streamlines logging and minimizes the amount of MLflow API calls required to capture run metadata, but it is comparatively less customizable than manual MLflow logging calls. The autologging configurations of supported libraries can be referenced [here](https://www.mlflow.org/docs/latest/tracking.html#automatic-logging) to weight whether this feature will effectively serve a given modeling use case.

### Model Signatures
Model signatures are an optional parameter to include within logged MLflow Models designed to enforce specific schemas regarding the included features and their associated data types that should be imputed into a model as well as the model's expected output. Model signatures can be both inferred directly by passing in example input and output data, as well as set manually by designating data input and output schema objects to establish the signature. Any data passed to a model loaded from MLflow Models that does not match the specified input structure in a model’s signature will raise an error.

Model signatures are most appropriate for circumstances such as production settings where verifying that unseen data follows a designated schema is key to ensuring the deployed model generates valid predictions. They are generally less helpful at earlier model development stages where experimentation around feature engineering may lead to violations of set model signatures.

## MLflow Model Registry

[MLflow’s Model Registry](https://www.mlflow.org/docs/latest/model-registry.html) provides a central location to track specific models across BLS use cases, particularly when they are moving beyond initial development and towards deployment. The Model Registry tab within MLflow’s UI records model lineage regarding which training runs produced the model, includes sections to add descriptions relevant to the concept of [model cards](https://arxiv.org/pdf/1810.03993.pdf), provides model versioning, and facilitates model stage transitions from development, staging, production, to archiving. The MLflow Tracking API offers a [range of methods](https://www.mlflow.org/docs/latest/model-registry.html#api-workflow) for tasks related to the Model Registry such as listing all registered models, renaming models, transitioning model versions and stages, and deleting models.

The model registry feature requires a database backend and is therefore not available within a default `mlflow ui` command prompt call on a local machine. The central BLS MLflow server is designed to be the primary location for registered models, so this requirement is not expected to be a significant pain point for individual users.

# MLflow at BLS

This repository features example files that explores how to use all four of MLflow's modules to best serve data science at BLS. It includes six Jupyter Notebooks and one MLproject example folder that each consider a different use case for MLflow as regarding various machine learning project goals, public BLS data sets, Python libraries, and employed algorithms.

## MLflow Intergration Example Notebooks & Projects

![example_viz.PNG](/imgs/example_viz.PNG)


*Visualizations logged as MLflow artifacts in the following example notebooks.* 

1. **sklearn_logreg_example_1.ipynb** is the example notebook designed to provide the most in-depth introduction to MLflow's Python API. It explores a multi-class multi-label classification task through a Scikit-learn logistic regression model on occupation work task text sourced from the Occupation Information Network (ONET) employment content modules. This notebook thoroughly explains the core modules and methods within the MLflow library and is therefore recommended reading for comprehensively understanding the platform for all of the additionally introduced use cases in the example notebooks that follow from this initial walkthrough.

2. **sklearn_rf_shap_example_2.ipynb** expands on the first Scikit-learn classification notebook to focus on how MLflow can support machine learning model interpretability and transparency regarding model predictions. This notebook highlights a Random Forest algorithm workflow on the multi-classification task of predicting bodily injury labels from narratives collected by the Mine Safety and Health Administration (MSHA). It additionally demonstrates how to integrate the SHAP (SHapley Additive exPlanation) library with MLflow to facilitate a more complete understanding of how the model generates its class assignments. 

3. **sklearn_regression_example_3.ipynb** illustrates a final Scikit-learn model workflow for a regression problem regarding predicting minutes spent on leisure time from a tabular data set sourced from the American Time Use Survey (ATUS). This notebook demonstrates how to record within MLflow a simultaneous cross-validation grid search of both three different regression model algorithms along with their affiliated hyperparameter search spaces simultaneously via the Scikit-learn Pipeline module along with techniques to incorperate feature selection. It additionally displays methods from the SHAP library building from the second example notebook as suited for regression models.

4. **hf_transformers_example_4.ipynb** expands from the same modeling goals and data set introduced in the first example notebook to consider how predicting ONET work activities from occupational task descriptions can be enhanced by state-of-the-art transformers neural network models via the Hugging Face library. Transformers are not currently supported by MLflow, so this example notebook delineates how to create a custom model loader class to successfully log and retrieve our created transformers model into the MLflow server. Furthermore, **hf_raytune_example_4_extended.ipynb** builds from this original notebook to delineate how the Ray Tune hyperparameter optimization library can be easily connected to report to MLflow to perform state-of-the-art Bayesian optimization searches on the hyperparameters associated within our established transformers model. 

5. **bertopic_unsupervised_example_5.ipynb** explores an unorthodox use case for MLflow regarding recording the created parameters and artifacts of an unsupervised BERTopic model tasked with discovering topics within the ONET occupational task data. While MLflow is originally designed for supervised learning applications, this example establishes how the platform can in fact greatly assist with ongoing challenges within unsupervised learning applications such as recording hyperparameter specifications and comparing discovered topics across fitted models through both tabular and visual artifacts. 

6. **mlprojects** is a directory within this repository that delineates how to create a MLflow Project model that can be ran from the command line and then launched as a live model that serves as a queryable REST API. This example folder describes the characteristics of MLflow's Project module as exemplified by specific file structures such as the MLProject file and a Python file designed for streamlined model runs that still accomodate hyperparameter specifications provided by the user.

## Tips for Effectively Using MLflow

### Experiments & Default Artifact Paths

Passing the `--default-artifact-root` parameter with your path of choice when activating the MLflow UI locally will redirect the default artifact logging directory behavior into a `./mlruns` folder. An important exception to this is with previously established experiments with already set artifact paths. You’ll need to create a new experiment if the original experiment was ran with a different artifact path than what `--default-artifact-root` is set to. 

### Starting & Ending Runs

Virtually all instances of a call to the MLflow API will start a model run if one isn't initialized already. It is therefore easy to accidentally start a run without realizing it when incorporating the MLflow API into a script. These unplanned runs must be ended with `mlflow.end_run` before a complete run intended for actual experiment logging can be subsequently launched, and can result in a much higher number of runs being logged than what was originally intended. This characteristic of the MLflow API encourages the use of local MLflow tracking servers within individual data science workspaces to ensure successful MLflow code integration before logging a run for cross-team reference within a remote server.

### Logging Single vs. Multiple Parameters, Metrics, and Artifacts
Parameters, metrics, and artifacts each have two Tracking API call versions designed for logging either single or multiple components. For example, both `log_parameter` and `log_metric` accepts a single string label and value pair, while `log_parameters`/`log_metrics` accepts a dictionary of label keys to multiple passed values. Equivalently, `log_artifact` will log a single file while `log_artifacts` will record all files within a provided directory path. You can refer to the `helpers.py` python file in the example folder of this repository for a helper function designed to create a metric dictionary for the specific example of the four outputs of Scikit-learn’s `precision_recall_f1_support`.

### Run & Artifact Garbage Collection

If you delete a run in the MLflow UI, the run will be removed from that experiment’s logger but its files will remain in its associated artifact directory. To remove these files, cd to the associated directory within a command prompt that contains the `mlruns` folder, activate the bls-mlflow conda environment or any equivalent environment you have MLflow installed within, and run `mlflow gc`. This will garbage collect all of the artifacts from the deleted runs.

### Comparing Metrics and Parameters Across Runs

The Compare feature within the MLflow UI is a great resource for visually understanding differences across model experimentation. The feature is based on separate model runs as dictated by calls to `mlflow.start_run` and `mlflow.end_run`. It is therefore not readily accessible for testing that occurs within the span of the same run, such as hyperparameter searches that are logged as parameters within one run. The starting and stopping of runs intended to be referenced in the Compare tab is therefore a relevant consideration when structuring MLflow API incorporation within scripts. A data scientist may want to nest tests within loops that activate multiple run starts and ends if they would like to use the Compare feature on particular aspects of their experimentation, but will also want to weight these decisions with potentially accumulating a high number of runs within one experiment.

### Storing Large Artifacts

MLflow is quite effective at storing the large amount of artifacts produced by advanced models such as neural network architectures. Libraries such as Hugging Face transformers and Pytorch commonly produce their models as binary files that are logged by MLflow, and they are liable to do so multiple times given their default configuration settings supporting actions such as model checkpointing after a designated number of epochs of training. These model files are usually multiple hundred MBs in size and checkpointing can lead to multiple different model copies being logged. This can quickly expand into unsubstaniable memory storage demands when logging multiple runs into MLflow each with an associated artifact directory retaining these large files. I therefore encourage data scientists who intend to use MLflow to support experimentation with these large models to consider memory management best practices such as by limiting model checkpointing or deleting model runs that are not relevant to share within team collaboration. 
