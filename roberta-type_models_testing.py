import pandas as pd
import numpy as np
import datasets
import evaluate
import torch
import random
import os 

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline
    )
from sklearn.metrics import classification_report
from transformers.pipelines.pt_utils import KeyDataset
from argparse import ArgumentParser

datasets.disable_progress_bar()
f1_metric = evaluate.load('f1')

# --------------- LABEL MAPPINGS ---------------

# Mappings for roberta-base and roberta-twittes (identical)
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {v:k for k, v in id2label.items()}

labels2twitter_roberta = {-1: 0, 0: 1, 1: 2}
twitter_roberta2labels = {v:k for k, v in labels2twitter_roberta.items()}

# Mappings for roberta-github
labels2gh_roberta = {-1: 2, 0: 0, 1: 1}
gh_roberta2labels = {v:k for k, v in labels2gh_roberta.items()}

# Mappings for Text2Text models
t2t_int2str = {-1: 'negative', 0: 'neutral', 1: 'positive'}
t2t_str2int = {v:k for k, v in t2t_int2str.items()}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # Using micro-f1, since we are utilizing balanced dataset
    return f1_metric.compute(predictions=predictions, references=labels, average='micro')

def set_seed(seed: None):
    """Set all seeds to make results reproducible (deterministic mode).
       When seed is None, disables deterministic mode.
    :param seed: an integer to your choosing
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


def eval(pipe, model, model_name, test_ds, print_report=False, return_pred_info=False):
    predicted_labels = []
    confidence_scores = []
    text_samples = []

    for i, out in enumerate(pipe(KeyDataset(test_ds, "text"))):
        label, score = out['label'], out['score']
        text = test_ds[i]['text']

        # str -> int, int -> int
        if 'twitter-roberta' in model_name or model_name == 'roberta-base':
            label = twitter_roberta2labels[model.config.label2id[label]]
            
        elif 'gh-roberta' in model_name:
            label = gh_roberta2labels[model.config.label2id[label]]
             
        predicted_labels.append(label)
        confidence_scores.append(round(score, 3))
        text_samples.append(text)

    if 'twitter-roberta' in model_name or model_name == 'roberta-base':
        true_labels = [twitter_roberta2labels[label] for label in test_ds['label']]
    
    elif 'gh-roberta' in model_name:
        true_labels = [gh_roberta2labels[label] for label in test_ds['label']]

    if return_pred_info:
        return {"texts": text_samples,
                "true_labels": true_labels,
                "predicted_labels": predicted_labels,
                "confidence scores": confidence_scores}

    if print_report:
        print(f'Results for {model_name}:\n')
        print(classification_report(true_labels, predicted_labels, digits=4))
    else:
        return classification_report(true_labels, predicted_labels, output_dict=True)


def load_data(train_path, 
              test_path, 
              few_shot_samples, 
              tokenizer, 
              max_len, 
              model_name, 
              stratify_seed=None, 
              text2text=False):

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_len)
       
    train_data, test_data = pd.read_csv(train_path), pd.read_csv(test_path)

    # Cast label to int to avoid mapping errors
    train_data.astype({'label': 'int32'})
    test_data.astype({'label': 'int32'})

    if text2text:
        train_data['label'] = train_data['label'].map(t2t_int2str)
        test_data['label'] = test_data['label'].map(t2t_int2str)
        
    else:
        if 'twitter-roberta' in model_name or model_name == 'roberta-base':
            train_data['label'] = train_data['label'].map(labels2twitter_roberta)
            test_data['label'] = test_data['label'].map(labels2twitter_roberta)

        elif 'gh-roberta' in model_name:
            train_data['label'] = train_data['label'].map(labels2gh_roberta)
            test_data['label'] = test_data['label'].map(labels2gh_roberta)

    if few_shot_samples > 0 and few_shot_samples < 200:
        # Perform startified sampling
        samples_per_label = int(few_shot_samples / 3) 
        train_data = train_data.groupby('label', group_keys=False)
        train_data = train_data.apply(lambda x: x.sample(samples_per_label, random_state=stratify_seed))
        train_data = train_data.sample(frac=1, random_state=stratify_seed)

    train_ds = datasets.Dataset.from_pandas(pd.DataFrame(data=train_data))
    test_ds = datasets.Dataset.from_pandas(pd.DataFrame(data=test_data))

    if text2text:
        return train_ds, test_ds

    train_ds = train_ds.map(preprocess_function, batched=True)
    test_ds = test_ds = test_ds.map(preprocess_function, batched=True)

    return train_ds, test_ds


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Provide model name / model repository from HuggingFace")
    parser.add_argument("--num_train_samples", type=int, help="Provide number of training samples (50, 100, 150, 202)")
    parser.add_argument("--train_data_csv", type=str, help="Provide path to train CSV file", default="data/train_data.csv")
    parser.add_argument("--test_data_csv", type=str, help="Provide path to test CSV file", default="data/test_data.csv")
    args = parser.parse_args()
    # model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    # model_name = 'marticampgin/gh-roberta-base-sentiment'
    # model_name = 'roberta-base'

    model_name = args.model_name
    few_shot_samples = args.num_train_samples
    train_path = args.train_data_csv
    test_path = args.test_data_csv

    default_single_seed = 77
    set_seed(default_single_seed)
    stratified_seeds = [55, 66, 77, 88, 99]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    reports = []

    if model_name == 'roberta-base':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3, id2label=id2label, label2id=label2id,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    MAX_LEN = tokenizer.max_model_input_sizes['roberta-base']  # 512
    
    # Fine-tune all model parameters
    for param in model.parameters():
        param.requires_grad = True

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=model_name + '-tested',
        learning_rate=2e-5,  # seems to be the most stable lr, but should also try lower values like 1e-5, potentially higher value 1e-4
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=4,  # can vary between 2-4
        weight_decay=0.01,  # could play with this parameter
        evaluation_strategy="epoch",
        logging_strategy = 'epoch',
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        seed=default_single_seed,
        disable_tqdm=True
    )
    
    # In case we want to test either zero-shot, or post full-finetuning performance
    if few_shot_samples in (0, 200):
        # Evaluate zero-shot performance (no finetuning)
        if few_shot_samples == 0:
            _, test_ds = load_data(train_path, test_path, few_shot_samples, tokenizer, MAX_LEN, model_name)

        # Evaluate performance one whole dataset (full finetuning)
        elif few_shot_samples == 200:
            train_ds, test_ds = load_data(train_path, test_path, few_shot_samples, tokenizer, MAX_LEN, model_name)

            trainer = Trainer(model=model,
                            args=training_args,
                            train_dataset=train_ds,
                            eval_dataset=test_ds,  
                            tokenizer=tokenizer,
                            data_collator=data_collator,
                            compute_metrics=compute_metrics)

            trainer.train()
        
        pipe = pipeline("text-classification", 
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        truncation=True,
                        max_length=MAX_LEN)
            
        eval(pipe, model, model_name, test_ds, print_report=True)

    else:
        num_of_runs = 5
        # Perform N number of runs for more robust results
        for i in range(num_of_runs):
            train_ds, test_ds = load_data(train_path, 
                                          test_path, 
                                          few_shot_samples, 
                                          tokenizer, 
                                          MAX_LEN,
                                          model_name, 
                                          stratified_seeds[i])
            
            # New model is trained and evaluated for each run
            trainer = Trainer(model=model,
                              args=training_args,
                              train_dataset=train_ds,
                              eval_dataset=test_ds,  
                              tokenizer=tokenizer,
                              data_collator=data_collator,
                              compute_metrics=compute_metrics)

            trainer.train()

            pipe = pipeline("text-classification", 
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            truncation=True,
                            max_length=MAX_LEN)
            
            # Collect results
            reports.append(eval(pipe, model, model_name, test_ds))

            # Re-init model
            if model_name == 'roberta-base':
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=3, id2label=id2label, label2id=label2id
                )
            else:
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
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
        
        print(f"\n{few_shot_samples}-shot learning results after {num_of_runs} runs, averaged for {model_name}", 
              end='\n------------------\n\n')
        
        for metric, result in avg_results.items():
            print(f'{metric:25}{result:.4f}')

        

if __name__ == '__main__':
    main()