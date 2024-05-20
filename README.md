## Repository for Master Thesis "Evaluating Large Language Models for Sentiment Analysis on IETF data".

Each folder contains Python `(.py)` or Jupyter Notebook `(.ipynb)` files with the code for each corresponding experiment conducted in the thesis. 

- [baseline_benchmarking](https://github.com/marticampgin/Master-Thesis/tree/main/baseline_benchmarking) contains `.ipynb` files with the code for initiating, training, and saving benchmark-models SVM and RNN. RNN from the experiments is also saved as a PyTorch `(.pt)` model.

- [chatgpt_benchmarking](https://github.com/marticampgin/Master-Thesis/tree/main/chatgpt_benchmarking) contains a `.ipynb` file with the code for establishing a connection to the ChatGPT model and testing it. This is done through OpenAI API. To run the code, a valid OpenAI access token is required. To create the token, log in to the OpenAI website, and open [Project API keys](https://platform.openai.com/api-keys) page.

- [flanT5_benchmarking](https://github.com/marticampgin/Master-Thesis/tree/main/flanT5_benchmarking) contains `.py` scripts with the code to conduct FLAN-T5 experiments, both for regular and quantized versions. Moreover, the folder contains a "pickle files" folder with several pickle `(.pkl)` files. This file contains data samples we downloaded during  the fine-tuning with a small number of sample experiments. This is done to ensure that when testing few-shot learning vs. fine-tuning, we utilize the same data samples. If you plan to run the code, make sure to include the folder in the same directory as the `.py` scripts.

- [roberta_benchmarking](https://github.com/marticampgin/Master-Thesis/tree/main/roberta_benchmarking) contains a single `.py` script to run experiments with all the RoBERTa models. 

- [data creation & exploration](https://github.com/marticampgin/Master-Thesis/tree/main/data%20creation%20%26%20exploration) contains `.py` scripts that scrape the names of active groups from the IETF website, compare them to filenames of in the archive of downloaded files, and extracts and transforms data to arrive at the dataset utilized in the thesis. A use case is demonstrated in the `.ipynb` file [obtaining_data.ipynb](https://github.com/marticampgin/Master-Thesis/blob/main/data%20creation%20%26%20exploration/obtaining_data.ipynb). The folder additionally contains a `.txt` file "short_messages_to_keep.txt", which is necessary for correct filtering of shorter messages. Note that to reproduce the steps conducted in the thesis, you will need to download a ZIP archive containing email archives and extract the folder inside into the same directory. You can download the archives [here](https://drive.google.com/file/d/1YP46kIW1kaMzOfrFfxOL7o-lszWHj7Ci/view?usp=drive_link).

- [data](https://github.com/marticampgin/Master-Thesis/tree/main/data) contains the labeled dataset in the form of several files: 2 CSV files with train & test data. Moreover, a `(.pkl)` file containing a dictionary of unlabeled data samples is also included. The labeled data samples represent a subset of the unlabeled data (in other words, is included).

- [requirements.txt](https://github.com/marticampgin/Master-Thesis/blob/main/requirements.txt) contains all the Python libraries we utilized to conduct this thesis. To install the libraries,
      - make sure that you are using Python version 3.11.5 or higher;
      - that you are in a correct virtual environment;
      - use command `pip install -r requirements.txt`



