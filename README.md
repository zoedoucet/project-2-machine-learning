## Overview
This project explores the Covertype dataset, in which forest cover types are classified based on attributes such as elevation, aspect, slope, hillshade, soil type, and more. The goal of the project is to understand the dataset, develop and tune a model, and evaluate its performance on a test dataset.
---

## Project Structure
The project starts with exploratory data analysis (EDA), followed by data cleaning and the development of a baseline model. The baseline model serves as a benchmark that every subsequent model aims to outperform. Next, four different models are developed and compared, and the two best-performing models are tuned. The final model is selected based on its performance. In the final stage, the model is evaluated and interpreted, saved, and reloaded to predict a few new samples.
---

## Requirements
The project uses:

* **Python**
* **Pandas** for data manipulation
* **Matplotlib** and **Seaborn** for visualisation
* **Jupyter Notebook** for exploratory analysis
* **Git/GitHub** for version control
* **Sklearn** to build and run the models
---
## How to Run

### 1. Clone the repository
Fork and clone the project directory.

### 2. Dowload the data
To download and process the data, run sections 2 and 6 of notebook 1.

Alternatively, run section 1 of notebook 2 or section 0 of notebook 3. In notebooks 2 and 3, a function automatically processes and cleans the data.

### 3. Run the notebooks
The notebooks can be run independently, depending on your interests. If you are only interested in the final model, you can go directly to notebook 3.
Run the notebook [01_eda_and_baseline](01_eda_and_baseline.ipynb) to get an insight on the exploratory data analysis and the baeline model.
Follow the model development process in notbook [02_pipelines_and_models](02_pipelines_and_models.ipynb), including the comparison of different models and hyperparameter tuning. Some models are computationally intensive and may take a long time to run.
Finally run notebook [03_evaluation_and_interpretation](03_evaluation_and_interpretation.ipynb) to take a look at its performance, its strengths and limitations, and how it can be used to make predictions on new samples.

### 4. Run the demo
Run the [gradio_app.py](gradio_app.py) code buy typing "py gradio_app.py" in your terminal.  The app will start locally, and a local URL will be generated. Click the link or copy it into your browser to open the app and try out the model. To stop the app, return to the terminal where it is running and press Ctrl + C. This will stop the local server and shut down the app.
---

## Dataset
For this project the Covertype dataset was used. This datset belongs to the UC Irvine Machine Learning Repository. I was donated on 31/07/1998 and created by Jock Blackard (https://doi.org/10.24432/C50K5N).

The dataset contains information about 30 x 30 meters cells of forest with infomation on the forest cover type and attributes such as elevation, aspect, slope, hillshade, soil type, and more.
Each row represents one 30 x 30 meter cell.

The raw dataset contains:

* **more than 500.000 observations**
* **54 columns**: 54 variables in total, including 10 quantitative features, 4 binary wilderness area features, 40 binary soil type features, and 1 target variable.

Information about
* the elevation, aspect and slope
* the wilderness area
* the soil type
* the hillshade at three different times of the day
* the distance to roads and fire points
* the horizontal and vertical distance to hydrology
---

## Research Questions

The goal of the project is to build a machine learning classification model that can **predict the forest cover type** based on these cartographic variables. The model could help anticipate which type of forest is likely to occur in a given area. The **target** variable is the forest **covertype**, while the remaining variables serve as input features. 

## Model Performance

The final model is a Random Forest classifier, selected based on its performance compared with the other models tested.

- **Accuracy:** 95.0%
- **Precision:** 0.940
- **Recall:** 0.897
- **F1-score:** 0.916

These metrics provide an overview of the model's predictive performance. The evaluation notebook explores its strengths, weaknesses, and classification errors in more detail.

---

## Limitations

The model predicts forest cover types based on cartographic and environmental variables rather than direct observations of vegetation. Some cover types can be difficult to distinguish due to similarities in their environmental characteristics.

---

## Acknowledgements

The project uses the Covertype dataset from the UCI Machine Learning Repository.

Blackard, J. (1998). *Covertype*. UCI Machine Learning Repository. https://doi.org/10.24432/C50K5N