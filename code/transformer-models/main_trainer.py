import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score, classification_report
import matplotlib.pyplot as plt
import scikitplot as skplt

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import Trainer, EarlyStoppingCallback


class Dataset:
    def __init__(self, dataset_path, text_col, labels_col):
        self.dataset_path = dataset_path
        self.text_col = text_col
        self.labels_col = labels_col
        
    def load_datapaths(self, data=None, train=None, valid=None, test=None, train_size=None):
        self.datapath = data
        self.train_datapath = train
        self.valid_datapath = valid
        self.test_datapath = test
        self.train_size = train_size
        
        self.__create_data_files__()
        
    # Create data files in the format required by HF Datasets
    def __create_data_files__(self):
        data_files = {}
        if (self.datapath == None):
            if (self.valid_datapath != None):
                data_files['valid'] = self.valid_datapath
        
            if (self.test_datapath != None):
                data_files['test'] = self.test_datapath
        
            if (self.train_datapath != None):
                data_files['train'] = self.train_datapath
                
        else:
            data_files['train'] = self.datapath
        
        if len(data_files) > 0:
            self.data_files = data_files
            print('Data files created.')
            print('data_files =', self.data_files)
            
            self.__create_hf_dataset__()
        else:
            print('ERROR: Datapaths not loaded. Call method "load_datapaths()" first !')
    
    # Creates dataset in Hugging Face Dataset format
    def __create_hf_dataset__(self):
        self.hf_ds = load_dataset(path=self.dataset_path, data_files=self.data_files)
        print('HF Datasets created.')
        print('HF Dataset =', self.hf_ds)
        
        # Preprocess HF dataset
        self.__preprocess_hf_dataset__()
    
    def __preprocess_hf_dataset__(self):
        # remove unwanted columns
        all_cols = set(self.hf_ds['train'].column_names)
        needed_cols = set([self.text_col, self.labels_col])
        cols_to_remove = list(all_cols - needed_cols)
        self.hf_ds = self.hf_ds.remove_columns(cols_to_remove)
        print('Removed columns:', cols_to_remove)

        # Rename to standard column names
        cols_to_rename = {self.text_col: 'text', self.labels_col: 'labels'}
        self.hf_ds = self.hf_ds.rename_columns(cols_to_rename)
        print('Renamed columns to standard names.')
        self.text_col = 'text'
        self.labels_col = 'labels'
        
        print('HF Dataset =', self.hf_ds)
        
        # Class Labels Encoding
        self.hf_ds = self.hf_ds.class_encode_column(self.labels_col)
        print(f'Feature "{self.labels_col}" encoded to ClassLabel.')
        
        # Train Valid Test Split
        if self.train_size != None:
            self.__train_test_split__()
            
        
    def __train_test_split__(self):
        if (self.train_size != None):
            assert 0.9 >= self.train_size > 0
            if len(self.hf_ds) > 1:
                print("ERROR: There should be only one key in Dataset !!!")
            else:
                # Start splitting
                ds = self.hf_ds['train'].train_test_split(train_size=self.train_size, stratify_by_column=self.labels_col, seed=0)

                ds_valid = ds['test'].train_test_split(test_size=0.5, stratify_by_column=self.labels_col, seed=0)
                ds['valid'] = ds_valid['train']
                ds['test'] = ds_valid['test']
                
                self.hf_ds = ds
                print("Dataset split with Train size: ", self.train_size)
                print('HF Dataset =', self.hf_ds)
    
    def get_num_labels(self):
        return self.hf_ds['train'].features['labels'].num_classes
    
    def get_label2id_and_id2label(self):
        category_names = self.hf_ds['train'].features['labels'].names
        id2label = label2id = {}

        for i in range(0, len(category_names)):
            label2id[category_names[i]] = i
    
        id2label = {str(value):key for key, value in label2id.items()}
        
        return label2id, id2label
    
    def set_tokenized_dataset(self, dataset):
        self.tokenized_dataset = dataset

        
    
    # For printing dataset object
    def __str__(self):
        print('Dataset path:', self.dataset_path)
        print()
        print('Data path:', self.datapath)
        print('Train Data path:', self.train_datapath)
        print('Valid Data path:', self.valid_datapath)
        print('Test Data path:', self.test_datapath)
        print()
        print('Data files:', self.data_files)
        
        return "\nDataset Summary"


class TransformerModel:

    def __init__(self, dataset, arch, model_id, tokenizer_id, max_seq_length=512, checkpoint=None, tds_path=None):
        self.dataset = dataset
        self.arch = arch
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.max_seq_length = max_seq_length

        # Check for checkpoint
        if checkpoint != None:
            self.model_id = checkpoint
            self.tokenizer_id = checkpoint

        # Load Model
        self.model = self.load_model()
        
        # Load Tokenizer and Tokenize dataset
        tokenizer, tokenized_dataset = self.tokenize_dataset(tds_path)
        self.tokenizer = tokenizer
        self.dataset.set_tokenized_dataset(tokenized_dataset)
        
        # Add Data Collator
        self.data_collator = DataCollatorWithPadding(tokenizer)
        
    def load_model(self):
        print(f'Loading {self.arch} model: {self.model_id}...')
        label2id, id2label = self.dataset.get_label2id_and_id2label()
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id, 
                                                                        num_labels=self.dataset.get_num_labels(), 
                                                                        label2id=label2id, 
                                                                        id2label=id2label)
        print('Model loaded !')
        return model
    
    def load_tokenizer(self):
        print(f'Loading pretrained tokenizer: {self.tokenizer_id}...') 
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, 
                                                       strip_accents=False, 
                                                       clean_text=False)
        return tokenizer

    def tokenize_dataset(self, tds_path):
        tokenizer = self.load_tokenizer()
        
        def tokenize_function(data):
            return tokenizer(data['text'], truncation=True, padding=True, max_length=self.max_seq_length)
        
        if tds_path is None:
            print('Tokenizing dataset...')
            tokenized_dataset = self.dataset.hf_ds.map(tokenize_function, batched=True, num_proc=4)
            tokenized_dataset = tokenized_dataset.remove_columns('text')
            tokenized_dataset = tokenized_dataset.with_format('torch')
        else:
            print('Loading tokenized dataset from path...')
            with open(tds_path, mode='rb') as f:
                tokenized_dataset = pickle.load(f)
        
        print('Tokenized dataset: \n', tokenized_dataset)
        
        return tokenizer, tokenized_dataset

class NepDistilBERT(TransformerModel):
    arch = 'DistilBERT'
    model_id = 'Sakonii/distilbert-base-nepali'
    tokenizer_id = model_id

    def __init__(self, dataset, max_seq_length, checkpoint=None, tds_path=None):
        super().__init__(dataset, self.arch, self.model_id, self.tokenizer_id, max_seq_length, checkpoint, tds_path)


class RajanBERT(TransformerModel):
    arch = 'BERT'
    model_id = 'Rajan/NepaliBERT'
    tokenizer_id = model_id

    def __init__(self, dataset, max_seq_length, checkpoint=None, tds_path=None):
        super().__init__(dataset, self.arch, self.model_id, self.tokenizer_id, max_seq_length, checkpoint, tds_path)

class ShushantBERT(TransformerModel):
    arch = 'ShushantBERT'
    model_id = 'Shushant/nepaliBERT'
    tokenizer_id = model_id

    def __init__(self, dataset, max_seq_length, checkpoint=None, tds_path=None):
        super().__init__(dataset, self.arch, self.model_id, self.tokenizer_id, max_seq_length, checkpoint, tds_path)

class NepRoBERTa(TransformerModel):
    arch = 'RoBERTa'
    model_id = 'amitness/nepbert'
    tokenizer_id = model_id

    def __init__(self, dataset, max_seq_length, checkpoint=None, tds_path=None):
        super().__init__(dataset, self.arch, self.model_id, self.tokenizer_id, max_seq_length, checkpoint, tds_path)

class NepDeBERTa(TransformerModel):
    arch = 'DeBERTa'
    model_id = 'Sakonii/deberta-base-nepali'
    tokenizer_id = model_id

    def __init__(self, dataset, max_seq_length, checkpoint=None, tds_path=None):
        super().__init__(dataset, self.arch, self.model_id, self.tokenizer_id, max_seq_length, checkpoint, tds_path)

class MBERT(TransformerModel):
    arch = 'mBERT'
    model_id = 'bert-base-multilingual-uncased'
    tokenizer_id = model_id

    def __init__(self, dataset, max_seq_length, checkpoint=None, tds_path=None):
        super().__init__(dataset, self.arch, self.model_id, self.tokenizer_id, max_seq_length, checkpoint, tds_path)

class XLMRoBERTa(TransformerModel):
    arch = 'XLMRoBERTa'
    model_id = 'xlm-roberta-base'
    tokenizer_id = model_id

    def __init__(self, dataset, max_seq_length, checkpoint=None, tds_path=None):
        super().__init__(dataset, self.arch, self.model_id, self.tokenizer_id, max_seq_length, checkpoint, tds_path)
    
class HindiRoBERTa(TransformerModel):
    arch = 'HindiRoBERTa'
    model_id = 'flax-community/roberta-hindi'
    tokenizer_id = model_id

    def __init__(self, dataset, max_seq_length, checkpoint=None, tds_path=None):
        super().__init__(dataset, self.arch, self.model_id, self.tokenizer_id, max_seq_length, checkpoint, tds_path)


class ModelTrainer:
    def __init__(self, training_id, hp, t_model, report_to, save_limit=1):
        self.training_id = training_id
        self.hp = hp
        self.t_model = t_model
        self.trainer = None
        
        self.training_args = self.__setup_training_args(save_limit, report_to)
        
        self.trainer = self.__setup_trainer()
        
    def __setup_training_args(self, save_limit, report_to):
        
        def parent(path):
            return '/'.join(path.split('/')[:-1])

        # Define training arguments
        training_args = TrainingArguments(output_dir=os.path.join(parent(parent(self.t_model.dataset.dataset_path)), 'models', self.training_id, 'checkpoints'),
                                          logging_dir=os.path.join(parent(parent(self.t_model.dataset.dataset_path)), 'models', self.training_id, 'logs'),
                                            save_strategy='steps',
                                            evaluation_strategy='steps',
                                            num_train_epochs=self.hp.epochs,
                                            per_device_train_batch_size=self.hp.train_bs,
                                            per_device_eval_batch_size=self.hp.eval_bs, 
                                            learning_rate=self.hp.lr, 
                                            weight_decay=self.hp.weight_decay,
                                            save_total_limit=save_limit,
                                            save_steps=self.hp.steps,
                                            eval_steps=self.hp.steps,
                                            logging_steps = self.hp.steps,
                                            load_best_model_at_end=True,
                                            do_eval=True,
                                            report_to=report_to
                                          )
        self.training_args = training_args
        print('Training arguments loaded !')
        return training_args
        
    def __setup_trainer(self):
        
        def compute_metrics(p):
            pred, labels = p
            pred = np.argmax(pred, axis=1)

            accuracy = accuracy_score(y_true=labels, y_pred=pred)
            balanced_accuracy = balanced_accuracy_score(y_true=labels, y_pred=pred) 

            recall_weighted = recall_score(y_true=labels, y_pred=pred, average='weighted')    
            precision_weighted = precision_score(y_true=labels, y_pred=pred, average='weighted')    
            f1_weighted = f1_score(y_true=labels, y_pred=pred, average='weighted')

            recall_macro = recall_score(y_true=labels, y_pred=pred, average='macro')    
            precision_macro = precision_score(y_true=labels, y_pred=pred, average='macro')    
            f1_macro = f1_score(y_true=labels, y_pred=pred, average='macro')

            return {
                'accuracy': accuracy, 
                'balanced_accuracy': balanced_accuracy, 
                'precision_weighted': precision_weighted, 
                'recall_weighted': recall_weighted, 
                'f1_weighted':f1_weighted,
                'precision_macro': precision_macro, 
                'recall_macro': recall_macro, 
                'f1_macro': f1_macro
                }
    
        # Define trainer
        trainer = Trainer(
            self.t_model.model,
            self.training_args,
            train_dataset=self.t_model.dataset.tokenized_dataset['train'],
            eval_dataset=self.t_model.dataset.tokenized_dataset['valid'],
            data_collator=self.t_model.data_collator,
            tokenizer=self.t_model.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.hp.patience)]
        )
        
        return trainer
        print('Trainer loaded. Ready to train !')
        
    def train_and_evaluate(self):
      print(self.hp)
      self.trainer.train()

    
    def test(self, show_classification_report=True, show_confusion_matrix=True):
        self.y_true, self.y_pred, self.test_metrics, self.prediction_result = self.__do_predictions()
        
        if show_classification_report:
            self.show_classification_report()
        
        if show_confusion_matrix:
            self.show_confusion_matrix()
            self.show_normalized_confusion_matrix()
            

   
    def __do_predictions(self):
        y_true = self.t_model.dataset.tokenized_dataset['test']['labels']
        print('y_true loaded from test dataset!')

        print('Starting predictions...')
        prediction_result = self.trainer.predict(self.t_model.dataset.tokenized_dataset['test'])
        raw_preds, _, test_metrics = prediction_result
        print('Predictions complete.')

        y_pred = [np.argmax(prediction) for prediction in raw_preds]
        print('y_pred added !')

        return y_true, y_pred, test_metrics, prediction_result

    def show_classification_report(self):
        print('Classification Report:\n')
        print(classification_report(self.y_true, self.y_pred, target_names=self.t_model.model.config.label2id.keys()))

    def show_confusion_matrix(self, save_image=False):
        data = {'y_true': self.y_true, 'y_pred': self.y_pred}
        df = pd.DataFrame(data, columns=['y_true', 'y_pred'])

        labels_to_category = {}
        for key, value in self.t_model.model.config.id2label.items():
            labels_to_category[int(key)] = value

        df.y_true = df.y_true.map(labels_to_category)
        df.y_pred = df.y_pred.map(labels_to_category)

        confusion_matrix = pd.crosstab(df['y_true'], df['y_pred'], rownames=['Actual Category'], colnames=['Predicted Category'], margins=False)
        fig = plt.figure(figsize=(10, 10))

        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt='g')
        plt.title(f'{self.t_model.model_id}\nLR = {self.hp.lr}\nTest Accuracy = {round(self.test_metrics["test_accuracy"], 4)}\n')
        
        if save_image:
            plt.savefig(f"{self.t_model.arch}_{self.t_model.model_id}_{self.training_id}.png", bbox_inches='tight', dpi=100)
            
        plt.show()

    def show_normalized_confusion_matrix(self):
        idtolabel = self.t_model.model.config.id2label

        def get_categories_from_tensors(tensors):
            y = tensors
            if not isinstance(tensors, list):
                y = tensors.tolist()
            y = [str(value) for value in y]
            y = [idtolabel[value] for value in y]

            return y

        ytrue = get_categories_from_tensors(self.y_true)
        ypred = get_categories_from_tensors(self.y_pred)

        skplt.metrics.plot_confusion_matrix(ytrue, ypred, normalize=True, figsize=(10, 10), x_tick_rotation=90)

class Parameters:
    def __init__(self, lr=5e-05, epochs=5, steps=500, patience=3, weight_decay=0.0, train_bs=32, eval_bs=32):
        self.lr = lr
        self.epochs = epochs
        self.steps = steps
        self.patience = patience
        self.weight_decay = weight_decay
        self.train_bs = train_bs
        self.eval_bs = eval_bs
        
    def __str__(self):
        print(f'lr = {self.lr}')
        print(f'epochs = {self.epochs}')
        print(f'steps = {self.steps}')
        print(f'patience = {self.patience}')
        print(f'weight_decay = {self.weight_decay}')
        print(f'train_bs = {self.train_bs}')
        print(f'eval_bs = {self.eval_bs}')
        return ""