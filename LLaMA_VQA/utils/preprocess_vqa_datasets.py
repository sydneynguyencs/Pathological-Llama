import torch
import clip
import pandas as pd
from PIL import Image
import pickle
import torch

from tqdm import tqdm
import string
import numpy as np
from transformers import set_seed, GPT2Config, GPT2Tokenizer, AutoTokenizer

def isEglish(s):
    return s.isascii()

def punc(s):
    for c in string.punctuation:
        s=s.replace(c,"")
    return s.lower() 

import pickle
import numpy as np
from transformers import GPT2Tokenizer

def create_tensors(pkl_file, tok="llama"):
    # standardize answer ids across datasets and compute the maximum number of generated output tokens based on the train set
    tokenizer = None
    if tok == "llama":
        tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf") 
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

    q_tokens = []
    q_lens = []
    a_tokens = []
    a_lens = []
    for question in data['questions']:
        q_token = torch.tensor(tokenizer.encode(str(question),add_special_tokens=False))
        q_tokens.append(q_token)
        q_lens.append(len(q_token))
    for answer in data['answers']:
        a_token = torch.tensor(tokenizer.encode(str(answer),add_special_tokens=False))
        a_tokens.append(a_token)
        a_lens.append(len(a_token))
    
    data['max_seqs_len']=(int(np.mean(q_lens)+2*np.std(q_lens)),int(np.mean(a_lens)+2*np.std(a_lens)))
    data['q_tokens'] = q_tokens
    data['a_tokens'] = a_tokens

    with open(pkl_file, 'wb') as f:
        pickle.dump(data,f)


def update_classes(pkl_train, pkl_val, pkl_test, tok="llama"):
    # standardize answer ids across datasets and compute the maximum number of generated output tokens based on the train set
    tokenizer = None
    if tok == "llama":
        tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf") 
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    with open(pkl_train, 'rb') as f:
            data_train = pickle.load(f)
    with open(pkl_val, 'rb') as f:
            data_val = pickle.load(f)
    with open(pkl_test, 'rb') as f:
            data_test = pickle.load(f)
    
    cur_id = 0
    class_names_list = []
    class_ids_list = [[],[],[]]

    for i, data in enumerate([data_train,data_val,data_test]):
        
        for answer in data['answers']:
            if answer not in class_names_list:
                class_names_list.append(answer)
                class_ids_list[i].append(cur_id)
                cur_id+=1
            else:
                class_ids_list[i].append(class_names_list.index(answer))
    
    data_train['class_ids'] = class_ids_list[0]
    data_val['class_ids'] = class_ids_list[1]
    data_test['class_ids'] = class_ids_list[2]
    
    data_train['class_names'] = class_names_list
    data_val['class_names'] = class_names_list
    data_test['class_names'] = class_names_list


    q_lens = []
    a_lens = []
    for question in data_train['questions']:
        q_token = tokenizer.encode(question, add_special_tokens=False)
        q_lens.append(len(q_token))
    for answer in data_train['answers']:
        a_token = tokenizer.encode(str(answer), add_special_tokens=False)
        a_lens.append(len(a_token))
    
    data_train['max_seqs_len']=(int(np.mean(q_lens)+2*np.std(q_lens)),int(np.mean(a_lens)+2*np.std(a_lens)))
    data_val['max_seqs_len']=(int(np.mean(q_lens)+2*np.std(q_lens)),int(np.mean(a_lens)+2*np.std(a_lens)))
    data_test['max_seqs_len']=(int(np.mean(q_lens)+2*np.std(q_lens)),int(np.mean(a_lens)+2*np.std(a_lens)))
    
    with open(pkl_train, 'wb') as f:
        pickle.dump(data_train,f)
    with open(pkl_val, 'wb') as f:
        pickle.dump(data_val,f)
    with open(pkl_test, 'wb') as f:
        pickle.dump(data_test,f)


def preprocess_pathvqa(split, in_path, out_path, img_path, question_type="oa"):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    data =  pd.read_pickle('{}/{}/{}_qa.pkl'.format(in_path,split,split))
    print("%0d captions loaded from json " % len(data))
    all_img_prefixes = []
    img_ids = []
    img_paths = []
    all_questions = []
    all_answers = []
    img_dict = {} # {key: img_id, value: [question->[], answer->[], img_prefix->tensor, img_path->str]}
    for i in tqdm(range(len(data))): # iterate through all data
        d = data[i]

        condition = True
        if question_type == "oa":
            condition = d['answer'].lower() not in ["yes", "no"]
        elif question_type == "closed":
            condition = d['answer'].lower() in ["yes", "no"]
        else:
            condition = True

        if condition: 
            img_id = d["image"] # img_id: "test_xxx""
            filename = "{}/{}/{}.jpg".format(img_path,split,img_id) # image path # makes more sense if absolute path
            with torch.no_grad():
                # prefix: encoding of image -> tensor
                prefix_i = clip_model.encode_image(preprocess(Image.open(filename)).unsqueeze(0).to(device)).cpu() # torch.Size([1, 512])               
            if img_id not in img_dict.keys():
                img_dict[img_id] = [[d['question']],[d['answer']],prefix_i,filename]
            else:
                img_dict[img_id][0].append(d['question'])
                img_dict[img_id][1].append(d['answer'])
    # this dictionary is converted into a format that is suitable for the data loader. Each data point contains an 'img_id', that corresponds is the index of the corresponding
    for img_id, imgs in enumerate(img_dict.keys()):
        all_img_prefixes.append(img_dict[imgs][2]) # used to be 3
        for q in range(len(img_dict[imgs][0])):
            all_questions.append(img_dict[imgs][0][q])
            all_answers.append(img_dict[imgs][1][q])
            img_ids.append(img_id)
            img_paths.append(img_dict[imgs][3]) # used to be 4

    all_data = {"img_prefix": torch.cat(all_img_prefixes, dim=0), "img_ids": img_ids, "questions": all_questions,'answers': all_answers,'img_path': img_paths}
    # img_prefix has shape torch.Size([839, 512]) (n_images, length)
    with open(out_path, 'wb') as f:
        pickle.dump(all_data,f)
    print('Done')
    print("%0d embeddings saved " % len(all_img_prefixes))


if __name__=='__main__':
    # question_types = ["closed","oa"] 
    question_types = ["both"]
    splits = ['test','val','train']
    img_path = "/home/ubuntu/Documents/mtDGX/data/pvqa/images"
    pvqa_in_data_path = "/home/ubuntu/Documents/mtDGX/data/pvqa/qas"
    pvqa_out_data_path = "/home/ubuntu/Documents/mtDGX/data/pvqa/CLIPLLAMA"
    tokenizer = "llama2" # options: llama2, gpt2-xl

    for question_type in question_types:
        for split in splits:
            out_path = "{}/{}_{}.pkl".format(pvqa_out_data_path,question_type,split) 
            preprocess_pathvqa(split,pvqa_in_data_path,out_path,img_path,question_type)
            pkl_file = "{}/{}_{}.pkl".format(pvqa_out_data_path,question_type,split) 
            create_tensors(pkl_file, tok=tokenizer)
    question_type = "both"
    update_classes(f"/home/ubuntu/Documents/mtDGX/data/pvqa/CLIPLLAMA/{question_type}_test.pkl",f"/home/ubuntu/Documents/mtDGX/data/pvqa/CLIPLLAMA/{question_type}_val.pkl",f"/home/ubuntu/Documents/mtDGX/data/pvqa/CLIPLLAMA/{question_type}_train.pkl")

                    

"""
closed question_type:
test: 3391
train: 9806
val: 3135
total: 16'332

oa question_type:
test: 3370 
train: 9949
val: 3144
total: 16463

both question_type:
test: 6761
train: 19755
val: 6279
total: 32795

test total: 6761
train total: 19755
val total: 6279
"""
