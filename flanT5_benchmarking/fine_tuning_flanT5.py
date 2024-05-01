import numpy as np
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          DataCollatorForSeq2Seq)

from sklearn.metrics import classification_report
from tqdm import tqdm
from roberta_type_models_testing import load_data, set_seed
from argparse import ArgumentParser


def generate_sample(text):
  return f"""Perform Sentiment classification task.
Given the text assign a sentiment label from ['negative', 'positive', 'neutral'].
Return label only without any other text.

<text>: {text}
<sentiment>: """.strip()


def run_on_test(test_dataset, model, tokenizer, str2int):
    golden_labels = []
    predicted_labels = []

    test_texts, test_labels = test_dataset['text'], test_dataset['label']
    inputs = [generate_sample(text) for text in test_texts]

    # Running inference on 1 sample at a time to avoid OOM issue
    for i, input in enumerate(tqdm(inputs)):
      input = tokenizer(input, return_tensors='pt').to('cuda')
      output = model.generate(**input)

      golden_labels.append(str2int[test_labels[i]])
      predicted_labels.append(str2int[tokenizer.decode(output[0], skip_special_tokens=True)])

    return golden_labels, predicted_labels


def preprocess_datasets(tokenizer, train_ds, test_ds):
    def preprocess_function(examples):
        inputs = [generate_sample(text) for text in examples['text']]
        model_inputs = tokenizer(inputs, max_length=1024,  truncation=True)

        # The labels are tokenized outputs
        labels = tokenizer(text_target=examples['label'],
                            max_length=512,
                            truncation=True)

        model_inputs['labels'] = labels['input_ids']

        return model_inputs
    
    tokenized_train_dataset = train_ds.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_ds.map(preprocess_function, batched=True)
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(['text', 'label'])
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(['text', 'label'])
   
    return tokenized_train_dataset, tokenized_test_dataset


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Provide model name / model repository from HuggingFace")
    parser.add_argument("--num_train_samples", type=int, help="Provide number of training samples (48, 96, 144, 204)")
    parser.add_argument("--train_data_csv", type=str, help="Provide path to train CSV file", default="data/train_data.csv")
    parser.add_argument("--test_data_csv", type=str, help="Provide path to test CSV file", default="data/test_data.csv")
    args = parser.parse_args()

    model_name = args.model_name
    few_shot_samples = args.num_train_samples
    train_path = args.train_data_csv
    test_path = args.test_data_csv

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    default_single_seed = 77
    set_seed(default_single_seed)
    stratified_seeds = [71, 72, 73, 74, 75]
    
    int2str = {-1: 'negative', 0: 'neutral', 1: 'positive'}
    str2int = {v:k for k, v in int2str.items()}

    max_input_len = 1024

    reports = []
    
    training_args = Seq2SeqTrainingArguments(
            output_dir="./flan-t5",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            num_train_epochs=3,
            evaluation_strategy="epoch",
            logging_strategy = 'epoch',
            save_strategy="epoch",
            predict_with_generate=True,
            push_to_hub=False,
            load_best_model_at_end=True)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    
    if few_shot_samples in (0, 204):
        if few_shot_samples == 0:
            _, test_ds = load_data(train_path, test_path, 0, tokenizer, max_input_len, model_name)
            model.to('cuda')
            golden_labels, predicted_labels = run_on_test(test_ds, model, tokenizer, str2int)
            print(f'Results for {model_name} (0 samples):\n')
            print(classification_report(golden_labels, predicted_labels, digits=4))

        elif few_shot_samples == 204:
            train_ds, test_ds = load_data(train_path, test_path, 204, tokenizer, max_input_len, model_name)
            tokenized_train_dataset, tokenized_test_dataset = preprocess_datasets(tokenizer, train_ds, test_ds)

            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_test_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator)
        
            trainer.train()

            golden_labels, predicted_labels = run_on_test(test_ds, model, tokenizer, str2int)
            print(f'Results for {model_name} (204 samples):\n')
            print(classification_report(golden_labels, predicted_labels, digits=4))

    else:
        for i in range(5):  # number of runs for. the final result is the average of these runs
            train_ds, test_ds = load_data(train_path,
                                          test_path,
                                          few_shot_samples,
                                          tokenizer,
                                          max_input_len,
                                          model_name,
                                          stratified_seeds[i],
                                          save_train_set=True,
                                          run_number=i+1)
            
            tokenized_train_dataset, tokenized_test_dataset = preprocess_datasets(tokenizer, train_ds, test_ds) 

            model.to('cuda')
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_test_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator
            )
            
            trainer.train()

            golden_labels, predicted_labels = run_on_test(test_ds, model, tokenizer, str2int)
            reports.append(classification_report(golden_labels, predicted_labels, output_dict=True))
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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
        
        print(f"\n{few_shot_samples}-shot learning results after 5 runs, averaged for {model_name}", 
                end='\n------------------\n\n')
        
        for metric, result in avg_results.items():
            print(f'{metric:25}{result:.4f}')


if __name__ == "__main__":
   main()