# Probabilistic Suffix Prediction of Buisiness Processes Development Repository

## Probabilitic Suffix Prediction Framework
We predict a probability distribution of suffixes of business processes using our own U-ED-LSTM and MC Suffix Sampling Algorithm.


## Setting Up the Python Environment with Pipenv

This project uses `pipenv` on Linux devices, for managing Python dependencies. Follow the steps below to set up the virtual environment and install the necessary packages using the provided `Pipfile`.

### Prerequisites
Make sure you have Python and Pipenv installed.
1. **Install Pipenv and pyenv**:
    
    ```bash
    pip install pipenv
    ```

    ```bash
    curl https://pyenv.run | bash
    ```

### Setup Instructions
2. **Navigate to the Project Directory**: Open your terminal and navigate to the project directory where the `Pipfile` is located.
    
    ```bash
    cd path/to/your/project
    ```

3. **Create the Virtual Environment**: Run the following command to create the virtual environment and install the dependencies from the `Pipfile`:
    
    ```bash
    pipenv install
    ```

4. **Activate the Virtual Environment**: Once the environment is set up, activate it with:
    
    ```bash
    pipenv shell
    ```

5. **Run the Project**: You can now run the project within the virtual environment. Execute a Juyter notebook.


## Run the Probabilistic Suffix Prediction Framework: Train and Evaluate.

- **data**: This folder contains the raw datasets.

- **encoded_data**: Stores the preprocessed datasets, which are used as inputs for the U-ED-LSTM training.

- **evaluation_results**: Stores the evaluation resutls.

- **related_work**: Contains an excel sheet with detailed categorization of the literature found and discussed in Section 5 Related work of the paper.

- **src**: Contains the source code for the probabilistic suffix prediction framework:
    - ``src``contains a directory ``notebooks``. Only this directory contains executable files.
        - To pre-process a dataset: ``src/notebooks/loader_notebooks``. The pre-processed datasets are located in: ``encoded_data``.
        - To train an U-ED-LSTM on a preprocessed dataset: ``src/notebooks/training_variatinal dropout/xxx``. The trained models are located in the same directory.
        - To evaluate a model: ``src/notebooks/evaluation_metric_notebooks``.

### Training

There is already a trained version of the U-ED-LSTM for each dataset, located in the directory: ``src/notebooks/training_variational_dropout/xxx``. Be cautious when re-running the Jupyter notebook in this directory, as it will overwrite the existing model. To prevent this, rename the model file before re-running the notebook to train a new version of the U-ED-LSTM.

Additionally, in the directory ``src/notebooks/training_variational_dropout/xxx``, there is a runs directory that stores TensorBoard files for monitoring the training process.

### Evaluation


