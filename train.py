from transformers import (AutoTokenizer, AdamW, get_scheduler)
from model import T5PromptTuningLM
import pandas as pd
from datasets import Dataset

import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from datasets import Dataset as HDataset
from tqdm import tqdm

class NewsDataset(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.instances = []

        dataset = pd.read_csv(file_path)
        dataset.dropna(inplace=True)
        for line, label in tqdm(zip(dataset["sentence"].values, dataset["label"].values), total=len(dataset["label"].values)):
            instance = {
                        "sentence": line+"</s>",
                        "label": "LABEL: " + label+"</s>",
                       }
            self.instances.append(instance)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        return self.instances[i]


class T5Classification(pl.LightningModule):
    def __init__(self, model_name:str="google/flan-t5-xxl",
                 num_train_epochs:int=3,
                 weight_decay:float=0.01,
                 learning_rate:float=0.01,
                 num_warmup_steps:int=0,
                 n_prompt_tokens:int=40,
                 init_from_vocab:bool=True):
        super().__init__()
        self.num_train_epochs=num_train_epochs
        self.weight_decay=weight_decay
        self.learning_rate=learning_rate
        self.num_warmup_steps=num_warmup_steps
        self.max_train_steps=num_train_epochs
        self.n_prompt_tokens=n_prompt_tokens
        self.init_from_vocab=init_from_vocab
        self.model_name = model_name

        self.model = T5PromptTuningLM.from_pretrained(self.model_name,
                                                      device_map="auto",
                                                      load_in_8bit=True,
                                                      n_tokens=self.n_prompt_tokens,
                                                      initialize_from_vocab=self.init_from_vocab)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.train_loss = []
        self.val_loss = []
        self.test_loss = []

    def my_collate(self, batch):
        sentences = []
        labels = []

        for instance in batch:
            sentences.append(instance["sentence"])
            labels.append(instance["label"])

        sentences_batch = self.tokenizer(sentences, padding="max_length", max_length=2008, truncation=True, return_tensors="pt")
        sentences_batch = {key:val.to("cuda") for key, val in zip(sentences_batch.keys(), sentences_batch.values())}

        labels_batch = self.tokenizer(labels, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        labels_batch = {key:val.to("cuda") for key, val in zip(sentences_batch.keys(), sentences_batch.values())}
        
        return sentences_batch, labels_batch

    def forward(self, sentence, label):
        outputs = self.model(input_ids=sentence["input_ids"], attention_mask=sentence["attention_mask"],
                             decoder_input_ids=label["input_ids"], decoder_attention_mask=label["attention_mask"])
        loss, logits = outputs.loss, outputs.logits
        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, logits = self(batch)
        self.train_loss.append(loss.detach().cpu().numpy())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self(batch)
        self.val_loss.append(loss.detach().cpu().numpy())
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits = self(batch)
        self.test_loss.append(loss.detach().cpu().numpy())
        return loss

    def configure_optimizers(self):
        optimizer = AdamW([p for n, p in self.model.named_parameters() if p.requires_grad], lr=self.learning_rate)

        return optimizer


def cli_main():
    pl.seed_everything(1234)

    # data
    train_dataset = NewsDataset("train.csv")
    test_dataset = NewsDataset("test.csv")

    model = T5Classification()
    trainer = pl.Trainer(max_epochs=model.max_train_steps)

    train_loader = DataLoader(train_dataset, batch_size=16,  shuffle=True, collate_fn=model.my_collate, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4, shuffle=False, collate_fn=model.my_collate, pin_memory=True, drop_last=True )
    #val_loader = DataLoader(valid_dataset, batch_size=16, num_workers=4, shuffle=False, collate_fn=model.my_collate, pin_memory=True, drop_last=True)

    trainer.fit(model, train_loader)
    print("Saving prompt...")
    save_dir_path = "./soft_prompt"
    model.model.save_soft_prompt(save_dir_path)
    trainer.test(test_dataloaders=test_loader)

if __name__ == "__main__":
    cli_main()
