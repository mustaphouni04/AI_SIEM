# synthesis_project_UAB
The cybersecurity challenge of iDISC in the subject of Synthesis Project I.


An LSTM Autoencoder that can be used for detecting anomalies in web-server log data.


![Architecture](documentation/architecture2.png)

## Repository structure

`logs/`: This directory contains the dataset used in this project, a series of log files.

`utils/`: This directory contains the Python scripts for the project, experimentation and model.

`rules/`: This directory contains the heuristic rules for detecting anomalies.

`documentation/`: This directory contains all technical documentation given for the project.

## Usage
Experimentation can be done without training the model.
But first, clone the repo using: 

```git clone https://github.com/mustaphouni04/synthesis_project_UAB.git```

Second, using the provided environment.yml file create a conda environment with the required dependencies using: 

```conda env create --file environment.yml```


Use *get_data_in_csv.py* to format the logs in CSV format, call the function.
After calling it, have a resulting combined_logs.csv file, load it as a DataFrame and pass it to *utils/preprocessing2.py* final_preprocessing function.
Turn the resulting preprocessed DataFrame back to a csv file and name this csv *preplogs_model2.csv*.

That's it! Now you can experiment in *utils/testing_second_model.ipynb*.

Additionally, if you want to train the model launch in utils folder:

```python main_tbmV2.py```

Make sure to input in the command prompt ```wandb login --relogin``` beforehand and the preprocessed csv path in the python file.

If you have any problem obtaining the data or any other issue contact us at mustaphounii04@gmail.com or directly input an issue in this repo, we will provide you the CSV files and data preprocessed.


## Authors:
Mustapha El Aichouni,
Ivan Martin,
Jana Zhi Lasheras,
Meritxell Carvajal

