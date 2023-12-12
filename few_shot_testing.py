import pandas as pd
import numpy as np
import datasets
import evaluate
import torch

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

datasets.disable_progress_bar()

# --------------- LABEL MAPPINGS ---------------

# Roberta base, exactly the same as twitter-roberta
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {v:k for k, v in id2label.items()}

labels2twitter_roberta = {-1: 0, 0: 1, 1: 2}
twitter_roberta2labels = {v:k for k, v in labels2twitter_roberta.items()}

labels2gh_roberta = {-1: 2, 0: 0, 1: 1}
gh_roberta2labels = {v:k for k, v in labels2gh_roberta.items()}

f1_metric = evaluate.load('f1')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # Using micro-f1, since we are utilizing balanced dataset
    return f1_metric.compute(predictions=predictions, references=labels, average='micro')


def eval(pipe, model, model_name, test_ds, print_report=False):
    predicted_labels = []
    confidence_scores = []

    for out in pipe(KeyDataset(test_ds, "text")):
        label, score = out['label'], out['score']

        # str -> int, int -> int
        if 'twitter-roberta' in model_name or model_name == 'roberta-base':
            label = twitter_roberta2labels[model.config.label2id[label]]  
        elif 'gh-roberta' in model_name:
            label = gh_roberta2labels[model.config.label2id[label]]

        predicted_labels.append(label)
        confidence_scores.append(round(score, 3))


    if 'twitter-roberta' in model_name or model_name == 'roberta-base':
        true_labels = [twitter_roberta2labels[label] for label in test_ds['label']]
    
    elif 'gh-roberta' in model_name:
        true_labels = [gh_roberta2labels[label] for label in test_ds['label']]


    if print_report:
        print(classification_report(true_labels, predicted_labels, digits=4))
    else:
        return classification_report(true_labels, predicted_labels, output_dict=True)


def load_data(train_path, test_path, few_shot_samples, tokenizer, max_len, model_name, stratify_seed=None):

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_len)
       
    train_data, test_data = pd.read_csv(train_path), pd.read_csv(test_path)

    # Cast label to int to avoid mapping errors
    train_data.astype({'label': 'int32'})
    test_data.astype({'label': 'int32'})

    if 'twitter-roberta' in model_name or model_name == 'roberta-base':
        train_data['label'] = train_data['label'].map(labels2twitter_roberta)
        test_data['label'] = test_data['label'].map(labels2twitter_roberta)

    elif 'gh-roberta' in model_name:
        train_data['label'] = train_data['label'].map(labels2gh_roberta)
        test_data['label'] = test_data['label'].map(labels2gh_roberta)

    if few_shot_samples > 0 and few_shot_samples < 203:
        # Perform startified sampling
        samples_per_label = int(few_shot_samples / 3) 
        train_data = train_data.groupby('label', group_keys=False)
        train_data = train_data.apply(lambda x: x.sample(samples_per_label, random_state=stratify_seed))
        train_data = train_data.sample(frac=1, random_state=stratify_seed)

    train_ds = datasets.Dataset.from_pandas(pd.DataFrame(data=train_data))
    test_ds = datasets.Dataset.from_pandas(pd.DataFrame(data=test_data))

    train_ds = train_ds.map(preprocess_function, batched=True)
    test_ds = test_ds = test_ds.map(preprocess_function, batched=True)

    return train_ds, test_ds


def main():
    # model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    model_name = 'marticampgin/gh-roberta-base-sentiment'
    # model_name = 'roberta-base'

    if model_name == 'roberta-base':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3, id2label=id2label, label2id=label2id
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    few_shot_samples = 203  # 51, 102, 153, 203

    train_path = 'train_data.csv'
    test_path = 'test_data.csv'

    seed = 77
    num_of_runs = 3
    stratify_seeds = [77, 88, 99]

    assert len(stratify_seeds) == num_of_runs

    reports = []

    MAX_LEN = tokenizer.max_model_input_sizes['roberta-base']
    
    # Fine-tune all model parameters
    for param in model.parameters():
        param.requires_grad = True

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=model_name + '-tested',
        learning_rate=2e-5,  # seems to be the most stable lr
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,  # can vary between 2-4
        weight_decay=0.01,  # could play with this parameter
        evaluation_strategy="epoch",
        logging_strategy = 'epoch',
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        seed=seed,
        disable_tqdm=True
    )
    
    if few_shot_samples == 0:
        # Evaluate zero-shot performance (no finetuning)
        _, test_ds = load_data(train_path, test_path, 0, tokenizer, MAX_LEN, model_name)

        pipe = pipeline("text-classification", 
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        truncation=True,
                        max_length=MAX_LEN)
        
        eval(pipe, model, model_name, test_ds, print_report=True)

    elif few_shot_samples == 203:
        # Run once on the whole dataset
        train_ds, test_ds = load_data(train_path, test_path, 203, tokenizer, MAX_LEN, model_name)

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
        # Perform N number of runs for more robust results
        for i in range(num_of_runs):
            train_ds, test_ds = load_data(train_path, 
                                          test_path, 
                                          few_shot_samples, 
                                          tokenizer, 
                                          MAX_LEN,
                                          model_name, 
                                          stratify_seeds[i])
            
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
        
        print(f"\n{few_shot_samples}-shot learning results after {num_of_runs} runs, averaged", 
              end='\n------------------\n\n')
        
        for metric, result in avg_results.items():
            print(f'{metric:25}{result:.4f}')

        




#if __name__ == '__main__':
#    main()