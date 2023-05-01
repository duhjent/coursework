import torch
import numpy as np

class NewsDataset(torch.utils.data.Dataset):
    __sentiment_map = {'neutral': [0, 1, 0], 'negative': [0, 0, 1], 'positive': [1, 0, 0]}
    
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.X = np.array(dataframe.title)
        self.y = np.array(dataframe.sentiment)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        text = str(self.X[idx])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.__sentiment_map[self.y[idx]], dtype=torch.float)
        }
