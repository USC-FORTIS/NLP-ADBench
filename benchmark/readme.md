<!-- # BELOW DOES NOT WORK ANYMORE
# Feature Selection and Benchmarking Procedure

This document outlines the procedure for conducting feature selection using a BERT model and benchmarking OD algorithms. Please follow the steps below carefully to ensure a smooth and effective benchmarking process.

## Preliminary Step: Configuration

1. Open the `experiment_config.py` file.
2. Input the necessary configuration information, such as the dataset paths.

## Step 1: Feature Selection and Saving

- Execute the `feature_select_and_save.py` script.
  - This script utilizes a BERT model to extract features.
  - The preprocessing results (in numpy format) will be saved locally for convenient access.

## Step 2: Benchmarking (filling metrics to [benchmark_results.xlsx](https://docs.google.com/spreadsheets/d/1P3UkHfzzvQmNRUmXpmXqhPpo5l1zCjTj/edit?usp=sharing&ouid=118152924941030920870&rtpof=true&sd=true) [it is in the Benchmark folder on google drive] )

### Part A: Running Most Benchmarks

1. Open and execute the `benchmark_most_of.ipynb` notebook.
   - This notebook runs the majority of the benchmark tests.
   - Record the relevant metrics as they are generated.

### Part B: Individual Benchmark Execution

1. Execute each file containing "benchmark" in its name individually.
   - These files run specific algorithms 
   - Record the relevant metrics as they are generated.

### Alternative Approach: Parallel Execution (Caution Advised)

- You may also use scripts within the `script.py` folder to parallelly run all files that include "benchmark" in their names.
  - **Caution**: This approach is likely to cause system crashes unless you have a very powerful computer setup. Proceed with caution if you decide to attempt this method.

 -->

# Import data
Get data from the google drive link below and put it in the data folder.
https://drive.google.com/drive/folders/1NmPzfE8COhqFgO-CZ1S7dsCJznY66uwe

Place all downloaded data into the `./data` folder in the root directory of this project.

# Run the code
To process the data using either BERT or GPT features, run the following commands from the root directory of the project:
## BERT
````bash
python [your_script_name].py bert
````

## GPT
````bash
python [your_script_name].py gpt
````
