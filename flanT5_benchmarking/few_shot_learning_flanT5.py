from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from sklearn.metrics import classification_report
from argparse import ArgumentParser

import pandas as pd
import datasets
import pickle
import tqdm
import numpy as np
import warnings
import os 
warnings.filterwarnings("ignore")


def generate_prompt(test_sample, num_few_shot_samples, few_shot_data=None):
    
    task_description = """Perform Sentiment classification task.
Given the text assign a sentiment label from ['positive', 'negative', 'neutral'].
Return label only without any other text.\n"""

    for i in range(num_few_shot_samples + 1):
        if i != num_few_shot_samples:
            sample = few_shot_data[i]
            text, label = sample['text'], sample['label']

        if i == num_few_shot_samples:
            task_description += f"\n<text>: {test_sample}\n<sentiment>:"
        else:
            task_description += f"\n<text>: {text}\n<sentiment>: {label}\n"

    return task_description


def run_on_test(train_dataset, test_dataset, model, tokenizer, num_few_shot_samples, str2int, max_len=1024):
    golden_labels = []
    predicted_labels = []

    test_texts, test_labels = test_dataset['text'], test_dataset['label']
    inputs = [generate_prompt(test_text, num_few_shot_samples, train_dataset) for test_text in test_texts]
  
    # Running inference on 1 sample at a time to avoid OOM issue
    for i, input in enumerate(tqdm.tqdm(inputs)):
      input = tokenizer(input, return_tensors='pt', max_length=max_len).to('cuda')
      output = model.generate(**input)

      golden_labels.append(str2int[test_labels[i]])
      predicted_labels.append(str2int[tokenizer.decode(output[0], skip_special_tokens=True)])

    return golden_labels, predicted_labels


def load_test_data(test_path, t2t_int2str):
    test_data = pd.read_csv(test_path)

    # Cast label to int to avoid mapping errors
    test_data.astype({'label': 'int32'})

    test_data['label'] = test_data['label'].map(t2t_int2str)
    test_ds = datasets.Dataset.from_pandas(pd.DataFrame(data=test_data))

    return  test_ds


def load_pickle_data(pickle_path):
    train_dict = pickle.load(open(pickle_path, "rb"))
    train_ds = train_dict["train"]

    return train_ds


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Provide model name / model repository from HuggingFace")
    parser.add_argument("--test_data_folder", type=str, help="Provide folder with test data")
    parser.add_argument("--pickle_data_folder", type=str, help="Provide folder with pickle files")
    parser.add_argument("--is_zero_shot", type=str, help="Whether the test is zero shot ('y' / 'n')")
    
    args = parser.parse_args()

    SAMPLE_NUMBERS = [3, 6, 9]  
    RUN_NUMBERS = [1, 2, 3]

    # Mappings
    t2t_int2str = {-1: 'negative', 0: 'neutral', 1: 'positive'}
    t2t_str2int = {v:k for k, v in t2t_int2str.items()}

    data_path = args.test_data_folder
    model_name = args.model_name
    is_zero_shot = args.is_zero_shot
    pickle_data_folder = args.pickle_data_folder

    test_ds = load_test_data(os.path.join(data_path, "train_data.csv"), t2t_int2str)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if is_zero_shot == "y":
        zero_shot = True
    else:
        zero_shot = False

    if zero_shot:
        # Run zero-shot tests
        golden_labels, predicted_labels = run_on_test(None, test_ds, model, tokenizer, 0, t2t_str2int)
        print(classification_report(golden_labels, predicted_labels))
        return
    
    # Run few-shot tests
    for num in SAMPLE_NUMBERS:
        reports = []

        # Incerasing max. seq. len. depending on how many samples are in the prompt
        if num == 3:
          max_len = 1024
        elif num == 6:
          max_len = 1536
        else:
          max_len = 2048

        for run in RUN_NUMBERS:
            train_ds = load_pickle_data(os.path.join(pickle_data_folder, f"train_set_sample_num={num}_run_num={run}.pkl"))
            golden_labels, predicted_labels = run_on_test(train_ds, test_ds, model, tokenizer, num, t2t_str2int, max_len)
            reports.append(classification_report(golden_labels, predicted_labels, output_dict=True))

        negative_prec, negative_rec, negative_f1 = [], [], []
        positive_prec, positive_rec, positive_f1 = [], [], []
        neutral_prec, neutral_rec, neutral_f1 = [], [], []
        micro_f1, macro_f1 = [], []

        for report in reports:
            negative_prec.append(report['-1']['precision'])
            negative_rec.append(report['-1']['recall'])
            negative_f1.append(report['-1']['f1-score'])

            positive_prec.append(report['1']['precision'])
            positive_rec.append(report['1']['recall'])
            positive_f1.append(report['1']['f1-score'])

            neutral_prec.append(report['0']['precision'])
            neutral_rec.append(report['0']['recall'])
            neutral_f1.append(report['0']['f1-score'])

            macro_f1.append(report['macro avg']['f1-score'])
            micro_f1.append(report['weighted avg']['f1-score'])

        # Average collected results for each run 
        avg_results = {'Negative precison': np.mean(negative_prec),
                        'Negative recall': np.mean(negative_rec),
                        'Negative f1-score': np.mean(negative_f1),
                        'Netural precison': np.mean(neutral_prec),
                        'Netural recall': np.mean(neutral_rec),
                        'Netural f1-score': np.mean(neutral_f1),
                        'Positive precison': np.mean(positive_prec),
                        'Positive recall': np.mean(positive_rec),
                        'Positive f1-score': np.mean(positive_f1),
                        'Macro avg': np.mean(macro_f1), 
                        'Weighted avg': np.mean(micro_f1)}

        print(f"\n{num}-shot learning results after 3 runs, averaged for {model_name}", 
                end='\n------------------\n\n')
        
        for metric, result in avg_results.items():
            print(f'{metric:25}{result:.4f}')


if __name__ == "__main__":
    main()