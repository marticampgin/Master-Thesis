## Repository for Master Thesis "Evaluating Large Language Models for Sentiment Analysis on IETF data".

Each folder contains Python (.py) or Jupyter Notebook (.ipynb) files with the code for each corresponding experiment conducted in the thesis. 

- [baseline_benchmarking](https://github.com/marticampgin/Master-Thesis/tree/main/baseline_benchmarking) contains .ipynb files with the code for initiating, training, and saving benchmark-models SVM and RNN. RNN from the experiments is also saved as a PyTorch (.pt) model.

- [chatgpt_benchmarking](https://github.com/marticampgin/Master-Thesis/tree/main/chatgpt_benchmarking) contains a .ipynb file with the code for establishing a connection to the ChatGPT model and testing it. This is done through OpenAI API. In order to run the code, a valid OpenAI access token is required. To create the token, log in to the OpenAI website, and open [Project API keys](https://platform.openai.com/api-keys) page.

- [data creation & exploration](https://github.com/marticampgin/Master-Thesis/tree/main/data%20creation%20%26%20exploration) contains .py scripts that scrape the names of active groups from the IETF website, compare them to filenames of in the archive of downloaded files, and extracts and transforms data to arrive at the dataset utilized in the thesis. A use case is demonstrated in the .ipynb file [obtaining_data.ipynb](https://github.com/marticampgin/Master-Thesis/blob/main/data%20creation%20%26%20exploration/obtaining_data.ipynb). The folder additionally contains a .txt file "short_messages_to_keep.txt", which is necessary for correct filtering of shorter messages. 

