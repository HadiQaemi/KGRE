from typing import List
from transformers import AdamW
from .cl_model.model import Model
from .cl_model.custom_dataset import CustomDataset
from .cl_model.losses import ConLoss
import torch.nn as nn
import pandas as pd
import tqdm.notebook as tq
import os
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
import torch
import numpy

class Classify(nn.Module):

    def __init__(self, config):
        super(Classify, self).__init__()
        self.config = config

    def train_model(self, training_loader, model, optimizer):

        losses = []
        correct_predictions = 0
        num_samples = 0
        model.train()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        loop = tq.tqdm(enumerate(training_loader), total=len(training_loader),
                          leave=True, colour='steelblue')
        criterion = ConLoss(0.5, 0.1)
        for batch_idx, data in loop:
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids) # (batch,predict)=(32,8)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            predicts = torch.sigmoid(outputs['predicts']).cpu().detach().numpy().round()
            targets = targets.cpu().detach().numpy()
            correct_predictions += numpy.sum(predicts==targets)
            num_samples += targets.size

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        return model, float(correct_predictions)/num_samples, numpy.mean(losses)

    def forward(self):
        meta_dfs = pd.read_csv(os.path.join(self.config.DATA_DIR, "WhereClause", "train-data_whereClause_intermediary.csv"))
        meta_dfs['category'] = meta_dfs.relations.str.split(' ')
        meta_dfs['main_category'] = meta_dfs.category.apply(lambda x:[a.split('.')[0] for a in x])
        meta_dfs.drop(columns=['relations','category'], inplace=True)

        
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(meta_dfs.main_category)
        df = pd.concat([meta_dfs[['text']], pd.DataFrame(labels)], axis=1)
        df.columns = ['text'] + list(mlb.classes_)
        df_train = df[:3200]
        df_test = df[3200:]
        df_train = df_train
        df_valid = df_test
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        target_list = list(df.columns)
        target_list = target_list[1:]

        train_dataset = CustomDataset(df_train, tokenizer, self.config.MAX_LEN, target_list)
        valid_dataset = CustomDataset(df_valid, tokenizer, self.config.MAX_LEN, target_list)
        test_dataset = CustomDataset(df_test, tokenizer, self.config.MAX_LEN, target_list)
        
        # Data loaders
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=self.config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )

        val_data_loader = torch.utils.data.DataLoader(valid_dataset,
            batch_size=self.config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        test_data_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=self.config.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = Model()
        model.to(device)
        optimizer = AdamW(model.parameters(), lr = 1e-5)
        model, train_acc, train_loss = self.train_model(train_data_loader, model, optimizer)
        return model, train_acc, train_loss