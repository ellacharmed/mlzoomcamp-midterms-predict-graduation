# Students' Graduation Rate Prediction

DataTalksClub's [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/projects/README.md) Midterm Project

This repository is the submission for the above-mentioned course's `**Midterm Project**` for the 2023 cohort. This midterm is in between Module 6 Trees and Module 8 Deep Learning of a 10 modules approximately, a 4-month duration, *free* course run by @Alexey and his team.

## Table of Contents
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Students' Graduation Rate Prediction](#students-graduation-rate-prediction)
  - [Table of Contents](#table-of-contents)
  - [Problem description](#problem-description)
    - [Project Folder Structure](#project-folder-structure)
    - [About this dataset](#about-this-dataset)
    - [Data Dictionary](#data-dictionary)
    - [Domain area research](#domain-area-research)
  - [Installation](#installation)
  - [EDA and Feature Engineering](#eda-and-feature-engineering)
    - [Feature Engineering](#feature-engineering)
    - [Feature Selection](#feature-selection)
  - [Model Selection](#model-selection)
    - [Model training](#model-training)
  - [Exporting notebook to script](#exporting-notebook-to-script)
  - [Model deployment](#model-deployment)
  - [Reproducibility](#reproducibility)
  - [Dependency and environment management](#dependency-and-environment-management)
  - [Containerization](#containerization)
  - [Cloud deployment](#cloud-deployment)
    - [Input](#input)
    - [User Interface](#user-interface)
  - [Lessons Learnt](#lessons-learnt)
    - [Life-cycle of a Machine Learning Model](#life-cycle-of-a-machine-learning-model)
  - [Future Improvements](#future-improvements)
  - [Conclusions](#conclusions)
  - [References](#references)

<!-- /code_chunk_output -->



## Problem description

The objective of this project is to predict if a student is on track to graduate within 5 years of commencing college, using the features documented in the [EDA notebook](#data-dictionary) replicated below. As it is expensive to both the student *and* the school to drag a 4-year education to longer than 5 years and beyond. From the macroeconimics perspective, means that we are not replenshing an aging population with enough workers fast enough.

It is also meant to identify what features have an impact and which ones have no bearing on a student's performance and the probabilites of them graduating in 4 to 5 years.

The selection of this particular problem statement and this dataset is intentional. Keep the data and problem simple and focus on deploying a "completed" midterms according to the [Midterms criteria and scoring rubrik](https://docs.google.com/spreadsheets/d/e/2PACX-1vQCwqAtkjl07MTW-SxWUK9GUvMQ3Pv_fF8UadcuIYLgHa0PlNu9BRWtfLgivI8xSCncQs82HDwGXSm3/pubhtml).

Initially had a multiclass problem with many, many features and thousands of rows. I overestimated my time and skills in scoping & deploying such a project. The Minimum Viable Product of a baseline LogisticRegression pickled model showed too many roadblocks for deploying the web-service and so that dataset was swapped with this one.

This project's files have been organized as follows.

### Project Folder Structure
```bash
.
├── README.md                    -- this file
├── 
├── dv.pkl                       -- the processed and scaled data
├── model.pkl                    -- the final model used in web app    
├── Dockerfile                   -- file to use for building a Docker image
├── notebook_exploratory.ipynb   -- EDA code
├── notebook_model_tuning.ipynb  -- modeling experiments
├── Pipfile                      -- setup the virtual environment and 
├──                                 dependency management
├── Pipfile.lock                 -- used by pipenv 
├── /artifacts                   -- folder for generated images and other artifacts, ├──                                 used in this report and this project
├── /data                        -- houses our data files  
├── /src                         -- project's python scripts
|   ├── config.py                -- global constants
|   ├── data_loader.py           -- code to read data files
|   ├── data_preprocessor.py     -- code to handle preprocessing of data
|   ├── data_feature_builder.py  -- code to create new features
|   ├── predict.py               -- code for the web-service
|   ├── predict_test.py          -- code to test web-service
```


### About this dataset


dataset [from @ roycekimmons.com](http://roycekimmons.com/tools/generated_data/graduation_rate)

> This is a fictional dataset and should only be used for data science training purposes.
> 
> This data set includes graduation rates from a four-year college with a variety of student factors.

The key questions we want to address:

> 1. How effective are entrance requirements at predicting gpa and graduation rate?
> 1. What would be the best way to improve graduation rates?

> All data sets are generated on-the-fly. So, you can increase your n by downloading a data set multiple times and combining the files. (n=1000) per large sample download

Based on the above statement, I have downloaded 5x times and combined them into one csv file so that we can work with a larger sample size of 5000 observations, instead of just 1000 observations.

Target is `graduate_in_5years` 

- feature-engineered from 'years_to_graduate' as 
  - 0: 'Yes' ie 'years_to_graduate' < 6
  - 1: 'No' ie 'years_to_graduate' > 5 
- was set to 4 years initially until EDA reveals median is 5.


### Data Dictionary

| **Column Name**       | **Description**                               | **Range** | **Data Type** |
|-----------------------|-----------------------------------------------|-----------|---------------|
| 'act_composite_score' | The average of English, Mathematics, Reading, and Science tests. (Numerical) | 1-36 | int |
| 'sat_total_score'     | Sum of Reading & Writing, and Mathematics. (Numerical) | 400 to 1600 | int |
| 'parental_level_of_education' |  For one parent. (Categorical)   | | |
| 'parental_income' | Income for one parent, per annum. (Numerical) | | int |
| 'high_school_gpa' | Grade point average upon graduating high school. (Numerical) | 2.0 to 4.0 | float |
| 'college_gpa' | Grade point average upon graduating college. (Numerical) | 2.0 to 4.0| float |
| 'years_to_graduate' | `target` Number of years to graduate college. (Numerical) | | int  |

The data dictionary describes the purpose of each column and how the column names were converted to have '_' underscores instead of spaces and all in lowercase.

Certain features also have obvious minimum and maximum accepted ranges and they've been used to clip these columns to be within the min-max range as a way to treat the outlier values.

### Domain area research

The SAT and ACT are standardized tests widely used for college admissions in the United States.

**ACT**: The average of a student's scores on the four ACT tests: English, Mathematics, Reading, and Science. The score range for each of the four tests is 1-36. The composite score is reported as a whole number from 1 to 36, and is rounded to the nearest whole number.  

**SAT** : The SAT total score is the sum of the two section scores, Evidence-Based Reading and Writing, and Mathematics. Each section is scored on a scale of 200 to 800, so the total score can range from 400 to 1600.

**GPA** : answer from google bard to the question 'What is a passing GPA?'

> A passing GPA is the minimum GPA that a student must maintain in order to remain in good standing and graduate from college. The passing GPA varies from school to school, but it is typically between 2.0 and 2.5.
> 
> Some schools may have higher passing GPAs for certain programs or majors. For example, a student in an engineering program may need to maintain a GPA of at least 3.0 in order to graduate.
> 
> Students who fall below the passing GPA may be placed on academic probation. If they do not improve their grades, they may be suspended or expelled from school.
> 
> It is important to note that a passing GPA is not the same as a good GPA. Many colleges and universities use GPAs to determine admissions decisions, scholarships, and honors programs. A student who wants to be competitive in the college admissions process should aim for a GPA of at least 3.0.

## Installation

![](/artifacts/images/powertoys.png) ![](/artifacts/images/execute_groups.png)

Uses VS Code Extension: Jupyter PowerToys for Group tags to run cells in one-click of the mouse, so the tuning can be done separately by classifier-type.

TODO add binder or nbviewer button

You can replicate this repository by 

1. Pulling Docker image from [](https://hub.docker.com/r/) <br><br>or <br><br>


<details>
<summary>1. via Dockerhub</summary>

  - From the terminal, execute the following command to pull the image from Docker hub

    ```bash
    docker run -it -p 9696:9696 graduate_app:latest
    ```
</details>


<details>
<summary>2. via github</summary>

  - Clone the GIT repository using the Green button, via https or ssh.
    ```bash
    git clone https://github.com/
    ```

  - Create virtual environment with your environment manager of choice, and then activate it. A Pipfile is supplied as this project was created using the `pipenv` library.

    ```bash
    pip install pipenv
    pipenv install
    pipenv shell
    ```

- Launch Flask

    Execute command to start the Flask app on localhost for testing locally. Run code below from the root of the project.
    ```bash
    python src/predict.py
    ```

- Build Dockerfile 

    Alternatively, you could also access the app via local Docker Container using the supplied Dockerfile and run as shown. 
    ```bash
    docker build -t <image-name>:<tag> .
    docker tag <image-name>:<tag> <your-username>/<image-name>:1.0
    docker push <your-username>/<image-name>:1.0
    ```
    where
    `docker build -t midterms:1.0` informs docker to build using the -tag '1.0' for image called 'midterms'.

- Run Docker image

  Once the image is built, it will show up in the Docker Desktop app and you can `Run` and spin up a container from there or use the command below from the CLI.

    ```bash
    docker run -d -p=80:80 --name=<container-name> <username>/<image-name>
    ```
    where
    - `<container-name>` specifies container name of your choice instead of a randomly generated name
    - `<username>` on Docker hub, if image is pushed and published
    - `<image-name>` provided during the build step

</details>


## EDA and Feature Engineering


<details>
<summary>Findings from exploratory data analysis</summary>

This is a summary of some of the core findings from Exploratoray Data Analysis. More detailed analysis can be found in the [*notebook_exploratory.ipynb*](/notebook_exploratory.ipynb) notebook.

**Assumptions about the dataset**:

- not many features on socioeconomics circumstances, or location of college or where the students are from i.e. within the state, or out-of-state, or international students
- the parental data is assumed to be for just *one* parent
- students are those just finishing high-school and not working adults returning to pursue more education
- when students do not graduate beyond the 5th year, they were not given any ultimatums to "*shape up or else*" and allowed to get to the 10th year
- also not clear if there are students who take longer than 10 years and automatically labeled a "dropout" and thus, didn't earn their undergraduate degrees
- no data on whether students themselves are working part-time, or in multiple part-time jobs or studying full-time. Time spent on studying is a major factor in determining GPA for *most* average students, I would think.

**Findings**:

![](/artifacts/images/target.png)

- the target has slight imbalance but it is not extreme, at least for those in the train set
- more than any other feature, current gpa earned in college is the best indicator of years spent in college to earn that degree
- one's parents intelligence or lack of paper-qualifications has no bearing on one's own measure of finishing college

**Out-of-scope**:

Due to time constrains, some questions raised initially at outset of project was not investigated. These include the following, but is not exhaustive:

- might the data be augmented? as this is a fictional set, possibilities are endless
- if college_gpa has highest feature importance, can modeling be done with just this one feature?
- would the DictVecorizer or OrdinalEncoder be better suited for the feature 'parental_level_of_education'?
- might 'parental_income' being binned and changed as categorical improve model's metrics?

</details>

<details>Dimensionality reduction
<summary>Features Selection & Engineering</summary>

### Feature Engineering

After some consideration, it was decided to convert the target 'years_to_graduate' feature to binary, so that the project's aim is a Binary Classification problem. The full analysis of this feature can be seen in [analysing_target heading of the exploratory notebook](/notebook_exploratory.ipynb)

<details>target Pie chart
<summary>Expand for images</summary>

![](/artifacts/images/target_pie.png)
</details>

### Feature Selection

From the heatmap via correlation analysis and feature importances attribute, the conclusion was to drop 2 features: 'act_composite_score' & 'high_school_gpa'

<details>Correlation heatmap
<summary>Expand for images</summary>

![](/artifacts/images/heatmap.png)

![](/artifacts/images/cat_feature_impt.png)
</details>
</details> 

## Model Selection

The selected model is `HistGradientBoostingClassifier` as it has the highest AUC score among all; with the best parameters as provided by grid_search.best_params_:

### Model training
```python
best_params = {
  'min_samples_leaf': 2, 
  'max_leaf_nodes': 5, 
  'max_iter': 100, 
  'max_depth': 12, 
  'learning_rate': 0.1, 
  'early_stopping': True, 
  'class_weight': 'balanced'}
best_hist = HistGradientBoostingClassifier(**best_params)
```

<details>List of classifiers evaluated
<summary>Expand for images</summary>

![](artifacts/images/tuned_metrics.png)

</details>

<details>Confusion matrix to evaluate which model is finally chosen
<summary>Expand for images for Confusion matrix </summary>

| HistGradientBoost      | Catboost               |
| ---------------------- | ---------------------- |
| ![hist_cm](/artifacts/images/hist_cm.png) | ![cat_cm](/artifacts/images/cat_cm.png) |

- 

</details>

## Exporting notebook to script

## Model deployment

## Reproducibility

## Dependency and environment management

## Containerization

## Cloud deployment

### Input

- no web service deployed
- testing via Command Line Interface (CLI) only

### User Interface

- no web service deployed

## Lessons Learnt

Working on an End-to-end Machine Learning project with the aim to deploy it taught me a lot of things, mainly what gaps still exists in my skillset more than anything else. And how much there is to think about! Just look at the choices we need to decide on during the life-cycle of this project. First and foremost being what sort of problem do we want to tackle?

As the main aim is to learn deployment, my criteria for a dataset is :

- simple, not too many features
- straightforward problem statement: **'Prediction of Graduating in 5 Years**
- not requiring too much pre-processing and feature-engineering

As it is, I hit a roadblock on using Pipelines and had to revert back to trusty ole DictVectorizer from lessons.

The other factor is in not realizing how our features rely on CSS and other web-framework's UI/UX methodologies to receive inputs and validate said input. I would put greater emphasis on this in capstone first, ahead of hyperparameter tuning so I don't have to reiterate too much and become the obstacle like in this midterms.

### Life-cycle of a Machine Learning Model

- Choice of dataset and problem statement
- Choice of data columns to use in training model
- Choice of cleaning and processing strategies of missing/dirty/incomplete data
- Choice of model
- Choice of metrics
- Choice of web framework eg: Flask, FAstAPI, Streamlit, PythonAnywhere, and more
- Choice of web server on the cloud eg: AWS, GCP or Azure
- Cost of libraries in terms of skills and experience of team involved
- (out-of-scope this round) Choice of apps/tools for continuous delivery and monitoring of model performance after deployment
- Cost considerations of all the above i.e. : is open source available?

## Future Improvements

Areas to cover for next capstone. Make sure to improve on these areas, at the very least, while learning modules08-10.

1. Deploy to the web with an interface that accepts inputs matching features used and that is validated before read in as a dataframe from json data.
2. Fix all the plotting code(s).
3. Tidy up the scripts further using OOP principles.
4. Tidy up documentation and docstrings and comments - check for outdated and wrong/mismatched comments.
5. Utilize Pipelines and investigate how to overcome the problem faced in [tweet on Pipeline issue](https://x.com/ellacharm3d/status/1720461928399827239?s=20).
6. Unify plots' colour scheme with website for "branding" presentation, regardless if using plotly, matplotlib or seaborn. Does a cmap exist that is present in all 3 plotting libraries?
7. Deeper dive in xgboost, catboost and lightgbm as they have built-in features not found in sklearn's estimators. For example, catboost has its own randomized_search() method that has a nifty plotting feature. (*unless as I've felt the rnd_search code is convoluted and make execution unnecesarily over-complicated and longer than necessary*).
8. Is using auto-ml libraries a boon or a hindrance for learning? eg `pandas_profiling`, `lazypredict`, `pycaret`, etc. It makes things faster, but everything is behind a "black box" of *how did they do that?*


## Conclusions

- 

## References

Here are some of my favourite references in learning Data Science and Machine Learning, that I keep returning to over the course of my studies.

- Documentations from the various python packages
- On imbalanced datasets:
  - [A Gentle Introduction to Imbalanced Classification](https://machinelearningmastery.com/what-is-imbalanced-classification/)
- On hyperparameter tunings (and others):
  - [Inria's Machine learning in Python with scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/tuning/parameter_tuning_module_intro.html)
- On Documentation How-TOs:
  - simplified project structure based off of [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
  - [The Turing Way: A handbook for reproducible, ethical and collaborative research](https://the-turing-way.netlify.app/index.html)
  - [The Turing Way "Zero to Binder"](https://the-turing-way.netlify.app/communication/binder/zero-to-binder.html)
  - [AI Practitioner Handbook by AISG](https://connect.aisingapore.org/2023/03/introducing-the-ai-practitioner-handbook-a-comprehensive-guide-to-delivering-ai-projects-straight-from-ai-singapores-engineers/)
- On Machine Learning:
  - [The AIAP™ Field Guide (Version 4.0)](https://connect.aisingapore.org/aiap-field-guide/)

