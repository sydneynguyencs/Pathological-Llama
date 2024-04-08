from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.prefix_mappers import MLP

class VQAmedModel(nn.Module):

    def __init__(
        self,
        prefix_length=2,
        prefix_size=512,
        args=None
    ):
        super(VQAmedModel, self).__init__()
        self.model_type = args.model_type
        self.prefix_length = prefix_length

        if self.model_type in ["gpt2","gpt2-xl","microsoft/biogpt"]:
            self.peftmodel = AutoModelForCausalLM.from_pretrained(self.model_type,load_in_8bit=True,device_map='auto')
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_type)
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
            self.peftmodel = get_peft_model(self.peftmodel,peft_config) 
            self.base_model_embedding_size = self.peftmodel.transformer.wte.weight.shape[1] 
        if self.model_type in ["llama","llama2","tiny-llama"]:
            if self.model_type == "tiny-llama": # TinyLlama-1.1B-intermediate-step-1431k-3T # TinyLlama/TinyLlama-1.1B-Chat-v0.4
                self.peftmodel = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.1",load_in_8bit=True,device_map='auto')
            else:
                self.peftmodel = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf",load_in_8bit=True,device_map='auto')
            self.tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf") 
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
            self.peftmodel = get_peft_model(self.peftmodel,peft_config) 
            self.base_model_embedding_size = self.peftmodel.base_model.model.model.embed_tokens.weight.shape[1] 
        
        self.clip_project = MLP((
                prefix_size,
                (self.base_model_embedding_size * prefix_length) // 2,
                self.base_model_embedding_size * prefix_length,
                self.base_model_embedding_size * prefix_length))
    
    def forward(self, prefix, embedding, mask, q_len, task="generative"):
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.base_model_embedding_size) # torch.Size([30, 6, 2048]), torch.Size([16, 6, 2048])
        embedding = embedding.detach()

        for b in range(prefix_projections.shape[0]):
            # insert the visual prefix after the question 
            embedding[b,q_len[b]:q_len[b]+self.prefix_length] = prefix_projections[b] 

        outputs = self.peftmodel(inputs_embeds=embedding, attention_mask=mask)
        outputs = outputs.logits 
        return outputs
    
    def generate_embedding(self, prefix, tokens, mask, q_len):
        prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.base_model_embedding_size)#.to(self.device)
        if self.model_type=='microsoft/biogpt':
            embedding_txt = self.peftmodel.transformer.embed_tokens(tokens)
        elif self.model_type in ['gpt2','gpt2-xl']:
            embedding_txt = self.peftmodel.transformer.wte(tokens) 
        elif self.model_type in ["llama","llama2","tiny-llama"]:
            embedding_txt = self.peftmodel.model.model.embed_tokens(tokens) 

        embedding_txt[q_len:q_len+self.prefix_length,:] = prefix_projections
        return embedding_txt.view(1, tokens.size(0), -1)

    