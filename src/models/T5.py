import torch
import json
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration


class T5Model(pl.LightningModule):
    def __init__(self, model_name, max_len=500, num_beams=5):
        super().__init__()
        self.model_name = model_name
        self.max_len = max_len
        self.num_beams = num_beams
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.test_step_outputs = []
        self.save_hyperparameters()  # save parameters with pytorch-lightning
        self.pred_path = None

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, batch, train=True):
        if not train:
            ids = batch['input_ids'].squeeze(1)
            mask = batch['attention_mask'].squeeze(1)
            return self.model.generate(input_ids=ids, attention_mask=mask, max_length=self.max_len,
                                       num_beams=self.num_beams, early_stopping=True)

        return self.model(input_ids=batch['input_ids'].squeeze(1),
                          attention_mask=batch['attention_mask'].squeeze(1),
                          labels=batch['label'].squeeze(1)).loss

    def training_step(self, batch, batch_idx):
        train_loss = self.forward(batch)
        self.log("train_loss", train_loss, sync_dist=True, prog_bar=False, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        batch = {key: value.to(self.device) for key, value in batch.items()}
        val_loss = self.forward(batch)
        self.log("val_loss", val_loss, sync_dist=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        generated_ids = self.forward(batch, train=False)
        gen_texts = []
        for sam in generated_ids:
            gen_texts += [self.tokenizer.decode(sam.squeeze(), skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True)]
        self.test_step_outputs += gen_texts

    def on_test_epoch_end(self) -> None:
        with open(self.pred_path, 'w') as f:
            json.dump(self.test_step_outputs, f)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def set_pred_path(self, path):
        self.pred_path = path


class CustomDataSet(Dataset):
    def __init__(self, pairs, tokenizer, max_inp_len=512, max_out_len=512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_inp_len = max_inp_len
        self.max_out_len = max_out_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item_idx):
        source = self.tokenizer.batch_encode_plus([self.pairs[item_idx]['input']], max_length=self.max_inp_len,
                                                  padding='max_length', return_tensors='pt', truncation=True)
        input_ids, attention_mask = source['input_ids'], source['attention_mask']

        target_encoding = self.tokenizer(
            [self.pairs[item_idx]['output']], padding="max_length", max_length=self.max_out_len, truncation=True,
            return_tensors="pt")

        label = target_encoding.input_ids
        label_mask = target_encoding.attention_mask
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label, 'label_mask': label_mask}


class DataModule(pl.LightningDataModule):
    def __init__(self, train_pairs, val_pairs, test_pairs, tokenizer, train_bs, test_bs, val_bs):
        super().__init__()
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.test_pairs = test_pairs
        self.tokenizer = tokenizer
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.val_bs = val_bs
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self, stage=None):
        self.train_dataset = CustomDataSet(self.train_pairs, self.tokenizer)
        self.val_dataset = CustomDataSet(self.val_pairs, self.tokenizer)
        self.test_dataset = CustomDataSet(self.test_pairs, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_bs, shuffle=True, num_workers=16, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_bs, shuffle=False, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_bs, shuffle=False, num_workers=2)
