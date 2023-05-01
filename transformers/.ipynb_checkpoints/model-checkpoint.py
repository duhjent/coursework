import torch
from torch import nn
from transformers import DistilBertModel

class DistilBertClassifier(nn.Module):
    def __init__(self, preclass_dim=768, num_classes=3):
        super().__init__()
        self.preclass_dim = preclass_dim
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = nn.Linear(preclass_dim, preclass_dim)
        self.act = nn.ReLU()
        self.droput = nn.Dropout(.1)
        self.classifier = nn.Linear(preclass_dim, num_classes)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        out_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = out_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = self.act(pooler)
        pooler = self.droput(pooler)
        out = self.classifier(pooler)
        
        return out