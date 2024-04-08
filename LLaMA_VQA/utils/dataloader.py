import sys
import torch
from torch.utils.data import Dataset
import pickle


class medvqaDataset(Dataset):
    def __init__(self, path, tokenizer, split='train',like_test=False,prefix_length=2,question_type= 'oa'):
        super().__init__()
        data_path = f"{path}{question_type}_{split}.pkl"
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        sys.stdout.flush()
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.img_ids = data["img_ids"]
        self.img_prefixes = data["img_prefix"]
        self.q_tokens = data['q_tokens']
        self.questions = data['questions']
        self.answers = data['answers']
        self.a_tokens = data['a_tokens']
        # self.class_ids = data['class_ids']
        # self.class_names = data['class_names']

        self.max_seqs_len = data['max_seqs_len'] # (17, 13) (question, answer)
        self.train_setting = True if (split!='test'and like_test==False) else False
        self.prefix_len = prefix_length
        
    def __len__(self):
        return len(self.answers)

    def pad_sequences(self,index): # TODO: tokens and mask should have same shape
        m = [torch.tensor(self.tokenizer.encode('question: ', add_special_tokens=False)),torch.tensor(self.tokenizer.encode(' context:', add_special_tokens=False)),torch.tensor(self.tokenizer.encode('answer ', add_special_tokens=False)),torch.tensor(self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False))]
        m_mask = [torch.ones(len(self.tokenizer.encode('question: ', add_special_tokens=False))),torch.ones(len(self.tokenizer.encode(' context:', add_special_tokens=False))),torch.ones(len(self.tokenizer.encode('answer ', add_special_tokens=False))),torch.zeros(len(self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False)))]   # added len for end of text. TODO: why torch.zeros? 
        q_tokens = self.q_tokens[index]
        a_tokens = self.a_tokens[index]

        if self.train_setting:
            # construct the model input. The order is question, image, answer. During training the answer is masked. Any padding is placed on the right of the sequence. 
            # placeholder tokens are used on the location where the visual prefix will be inserted, with q_len indicating this location. 
            q_tokens,q_mask,leftover_tokens = self.make_padding(self.max_seqs_len[0],q_tokens,question=True)
            q_len = m[0].size(0) + q_tokens.size(0) + m[1].size(0) # 14
            a_tokens,a_mask,_ = self.make_padding(self.max_seqs_len[1],a_tokens,leftover_tokens=leftover_tokens) # both shape [25]
            if len((a_tokens==0).nonzero())!=0:
                pad_start = (a_tokens==0).nonzero()[0]
            else:
                pad_start=[]
            a_tokens = torch.cat((a_tokens,m[3])) if len(pad_start)==0 else torch.cat((a_tokens[:pad_start],m[3],a_tokens[pad_start:]))  # torch.Size([26]) a_tokens + eos_token
            q_tokens = torch.cat((m[0],q_tokens,m[1],torch.ones(self.prefix_len),m[2],a_tokens)) # torch.Size([47]) 1 + 12 + 1 + 6 + 1 + 25 + 1
            q_mask = torch.cat((m_mask[0],q_mask,m_mask[1],torch.ones(self.prefix_len),m_mask[2],a_mask,m_mask[3])) # torch.Size([47])
            return q_tokens, q_mask, q_len
        else:
            # in the test stage we do not have access to the answer, so we just load the question. 
            # since inference is not performed batch-wised we don't need to apply padding
            q_tokens,q_mask,_ = self.make_padding_test_setting(self.max_seqs_len[0],q_tokens)
            q_len = m[0].size(0) + q_tokens.size(0) + m[1].size(0)
            q_tokens = torch.cat((m[0],q_tokens,m[1],torch.ones(self.prefix_len),m[2]))
            q_mask = torch.cat((m_mask[0],q_mask,m_mask[1]))
            return q_tokens, q_mask, q_len 
    
    def make_padding(self, max_len, tokens, question=False, leftover_tokens=0):
        padding = max_len - tokens.size(0)
        if padding > 0:
            padding_tensor = torch.full((padding,), self.tokenizer.pad_token_id, dtype=tokens.dtype)
            if question:  # if question, no padding
                leftover_tokens = padding
                mask = torch.ones(tokens.size(0), dtype=torch.long)
            else:
                tokens = torch.cat((tokens, padding_tensor, torch.full((leftover_tokens,), self.tokenizer.pad_token_id, dtype=tokens.dtype)))
                mask = torch.zeros(max_len + leftover_tokens, dtype=torch.long)
        elif padding == 0:
            if question:
                mask = torch.ones(tokens.size(0) + leftover_tokens, dtype=torch.long)
            else:
                mask = torch.zeros(tokens.size(0) + leftover_tokens, dtype=torch.long)
                padding_tensor = torch.full((leftover_tokens,), self.tokenizer.pad_token_id, dtype=tokens.dtype)
                tokens = torch.cat((tokens, padding_tensor))
        elif padding < 0:
            if question:
                tokens = tokens[:max_len]
                mask = torch.ones(max_len, dtype=torch.long)
            else:
                tokens = torch.cat((tokens[:max_len], torch.full((leftover_tokens,), self.tokenizer.pad_token_id, dtype=tokens.dtype)))
                mask = torch.zeros(max_len + leftover_tokens, dtype=torch.long)
        return tokens, mask, leftover_tokens

    def make_padding_test_setting(self, max_len, tokens, do_padding=False):
        padding = max_len - len(tokens)
        if padding < 0:
            return tokens[:max_len], torch.ones(max_len), 0
        mask = torch.cat([torch.ones(len(tokens)), torch.zeros(padding)])
        if do_padding:
            pad_token = self.tokenizer.pad_token_id
            pad_tensor = torch.full((padding,), pad_token, dtype=tokens.dtype)
            tokens = torch.cat([tokens, pad_tensor])
        return tokens, mask, padding 

    def __getitem__(self, index):
        prefix = self.img_prefixes[self.img_ids[index]] 
        label = self.answers[index]
        binary_label = [1 if label == "yes" else 0][0]
        tokens, mask, q_len  = self.pad_sequences(index) # exp5 pad = eos
        return prefix, tokens, mask, q_len, binary_label
