import torch
from torch.utils.data.distributed import DistributedSampler
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50"
import time
import wandb
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import logging
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from evaluate import load
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from utils.dataloader import medvqaDataset
from models import VQAmedModel


class PVQAmodel(torch.nn.Module):
    def __init__(self, args={}, task="generative"):
        super(PVQAmodel, self).__init__()
        """
        Config
        """
        self.args = args
        self.device = self._configure_device()
        self.start_epoch = 0
        self.task = task
        """
        Model
        """
        self.model = self._build_model()
        """
        Data
        """
        self.train_dataset = self._build_dataset("train")
        self.dev_dataset = self._build_dataset("val")
        self.test_dataset = self._build_dataset("test")
        self.train_loader = self._dataloader("train")
        self.val_loader = self._dataloader("val")
        self.test_loader = self._dataloader("test")
    
    def _configure_device(self):
        """
        Configures the device for the model based on the distributed rank.
        Each GPU will be assigned to a different process in distributed training.
        """
        if torch.cuda.is_available() and self.args.gpus_per_node > 1:
            device = torch.device(f"cuda:{self.args.rank}")
        elif torch.cuda.is_available() and self.args.gpus_per_node == 1:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        # device = self.args.rank
        print(f"Configured {device}.")
        return device

    def _build_dataset(self, dataset_key, split=None):
        if dataset_key in ["train", "val"]:
            dataset =  medvqaDataset(path=self.args.dataset_path+self.args.dataset+'/',tokenizer=self.model.tokenizer,split=dataset_key,prefix_length=self.args.prefix_length,question_type=self.args.question_type,like_test=self.args.like_test)#,abl=args.ablation)
        if dataset_key in ["test"]:
            dataset =  medvqaDataset(path=self.args.dataset_path+self.args.dataset+'/',tokenizer=self.model.tokenizer,split=dataset_key,prefix_length=self.args.prefix_length,like_test=True,question_type=self.args.question_type)
        return dataset
    
    def _dataloader(self, dataset_key):
        if dataset_key == "train":
            dataset = self.train_dataset
        if dataset_key == "val":
            dataset = self.dev_dataset
        if dataset_key == "test":
            dataset = self.test_dataset
        sampler = DistributedSampler(dataset, num_replicas=self.args.world_size, rank=self.args.rank, shuffle=False, drop_last=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, pin_memory=False, num_workers=self.args.num_workers, drop_last=False, shuffle=False, sampler=sampler)
        return dataloader
 
    def _build_model(self):
        model = VQAmedModel(
            prefix_length=self.args.prefix_length,
            args=self.args
        )
        # load existing model
        model = self._load_model(model)
        model = model.to(self.device)#.to(self.args.rank) # rank instead of device?
        if self.args.gpus_per_node > 1:
            model = torch.nn.parallel.DistributedDataParallel(model)
        print("Model successfully loaded.")
        return model
    
    def _load_model(self, model):
        suffix = f"{self.args.notes}_prefixlength_{self.args.prefix_length}_batchsize_{self.args.batch_size}_lr_{self.args.lr}_mapping_{self.args.mapping_type}_seed_{self.args.seed}_setting_{self.args.setting}"
        self.args.out_dir = os.path.join(os.path.join(self.args.out_dir, self.args.model_type), suffix)
        if not os.path.exists(self.args.out_dir):
            os.makedirs(self.args.out_dir)

        checkpoint_files = [f for f in os.listdir(self.args.out_dir) if f.endswith('model.pt')]
        if not checkpoint_files:
            print("No model checkpoints found. Training from scratch.")
            return model

        max_epoch = max([int(f.split('_')[0]) for f in checkpoint_files])
        checkpoint_path = os.path.join(self.args.out_dir, f"{max_epoch}_model.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            # Load the state dictionary into the model
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded model from the checkpoint at epoch {max_epoch}.")
            self.start_epoch = int(max_epoch)
        else:
            print(f"Checkpoint file {checkpoint_path} does not exist. Training from scratch.")
        
        return model
    
    def _configure_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.args.lr) # TODO: .module

    def _configure_scheduler(self, optimizer):
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.args.epochs * len(self.train_loader),
        )

    """
    Training Loop
    """
    def train_model(self):

        logging.basicConfig(level=logging.INFO)
        if self.args.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project = "PathVQA-OpenEnd",
                notes = self.args.notes,
                # track hyperparameters and run metadata
                config = self.args
            )
        optimizer = self._configure_optimizer() 
        scheduler = self._configure_scheduler(optimizer)

        best_valid_loss = float("inf")
        counter = 0
        log_dict = {}
        end_epoch = self.start_epoch + self.args.epochs

        for epoch in range(self.start_epoch, end_epoch):
            self.train_loader.sampler.set_epoch(epoch) # Set epoch for DistributedSampler
            start_time = time.time()

            avg_loss = self._train_epoch(epoch, optimizer, scheduler)

            if self.task == "generative":
                avg_val_loss, avg_bleu = self.evaluate(epoch)
                # Save best model
                if avg_val_loss < best_valid_loss or epoch == self.args.epochs-1 or epoch%5==0:
                    best_valid_loss = avg_val_loss
                    self._save_model(epoch)

                scheduler.step()
                elapsed_time = time.time() - start_time
                logging.info(
                    "\t Epoch {}/{} \t BLEU={:.4f} \t Loss={:.4f} \t Val Loss={:.4f} \t Time={:.2f}s".format(
                        epoch + 1, end_epoch, avg_bleu, avg_loss, avg_val_loss, elapsed_time
                    )
                )
                log_dict = {"Epoch": epoch, "Avg BLEU": avg_bleu, "Avg Loss": avg_loss, "Avg Val Loss": avg_val_loss}

            if self.task == "categorical":
                avg_val_loss, accuracy, precision, recall, f1 = self.evaluate_model_categorical(epoch)

            if self.args.wandb:
                wandb.log(log_dict)
                if avg_val_loss > avg_loss:
                    counter += 1

    def _save_model(self, epoch):
        if not os.path.exists(self.args.out_dir):
            os.makedirs(self.args.out_dir)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) or isinstance(self.model, torch.nn.DataParallel):
            save_state_dict = self.model.module.state_dict()
        else:
            save_state_dict = self.model.state_dict()
        torch.save(save_state_dict, os.path.join(self.args.out_dir, f"{epoch+1}_model.pt"))
        print(f"Saved model at epoch {epoch+1}.")

    def _create_embeddings(self, tokens):
        embeddings = None
        with torch.no_grad():  
            if self.args.model_type=='microsoft/biogpt':
                embeddings = self.model.peftmodel.transformer.embed_tokens(tokens)
            elif self.args.model_type in ['gpt2','gpt2-xl']:
                embeddings = self.model.peftmodel.transformer.wte(tokens) 
            elif self.args.model_type in ["llama","llama2","tiny-llama"]:
                if self.args.task == "explainability":
                    embeddings = self.model.peftmodel.model.model.embed_tokens.embedding(tokens)
                else:
                    embeddings = self.model.peftmodel.model.model.embed_tokens(tokens)
        return embeddings

    def _train_epoch(self, epoch, optimizer, scheduler):  
        self.model.train()
        total_loss = 0.0

        # train dataloader
        for i, (prefix, tokens, mask, q_len, _) in enumerate(self.train_loader):
            prefix = prefix.to(self.device, dtype=torch.float32)
            tokens = tokens.to(self.device, dtype=torch.long)
            mask = mask.to(self.device, dtype=torch.long)
            q_len = q_len.to(self.device, dtype=torch.long)

            embeddings = self._create_embeddings(tokens)
            outputs = self.model(prefix, embeddings, mask, q_len, task=self.task)
            logits = outputs
            loss = self._compute_loss(logits, tokens, q_len)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            avg_loss = total_loss / (i+1)

        return avg_loss
    
    def _compute_loss(self, logits, tokens, q_len, labels=None):
        loss = 0.0
        total_loss = 0.0
        for b in range(logits.size(0)):
            if self.task == "generative":
                prefix_length = self.model.module.prefix_length if isinstance(self.model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)) else self.args.prefix_length
                condensed_tokens = tokens[b,q_len[b]+prefix_length+1:]
                condensed_logits = logits[b,q_len[b]+prefix_length:-1] 
                loss = torch.nn.functional.cross_entropy(condensed_logits.reshape(-1,logits.shape[-1]), condensed_tokens.flatten(), ignore_index=0)
            if self.task == "categorical":
                loss = torch.nn.functional.cross_entropy(logits, labels)
            total_loss += loss
        return total_loss / logits.size(0)
    

    def evaluate(self, epoch=0, mode="eval"):
        self.model.eval()
        self.model.to(self.device)
        dataloader = self.test_loader if mode == "test" else self.val_loader
        total_loss, total_bleu, total_samples = 0.0, 0.0, 0

        for i, (prefix, tokens, mask, q_len, _) in enumerate(dataloader): 
            torch.cuda.empty_cache()
            prefix = prefix.to(self.device, dtype=torch.float32)
            tokens = tokens.to(self.device, dtype=torch.long)
            mask = mask.to(self.device, dtype=torch.long)
            q_len = q_len.to(self.device, dtype=torch.long)

            with torch.no_grad():
                embeddings = self._create_embeddings(tokens)
                outputs = self.model(prefix, embeddings, mask, q_len, task=self.task)
                generated_tokens = outputs.argmax(dim=-1)
                loss = self._compute_loss(outputs, tokens, q_len)
                total_loss += loss.item()

            batch_bleu_scores = self._compute_batch_bleu(generated_tokens, tokens, q_len)
            total_bleu += sum(batch_bleu_scores)
            total_samples += prefix.shape[0]

        print(f"Val Epoch: {epoch} Loss: {total_loss / (i + 1):.4f} BLEU: {total_bleu / total_samples:.4f}")

        avg_val_loss = total_loss / len(dataloader)
        avg_bleu = total_bleu / total_samples
        return avg_val_loss, avg_bleu

    def _compute_batch_bleu(self, generated_tokens, reference_tokens, q_len):
        bleu_scores = []
        for j in range(generated_tokens.shape[0]):
            ref_tokens = self._get_reference_tokens(reference_tokens[j], q_len[j])
            gen_tokens = generated_tokens[j].tolist()
            bleu = sentence_bleu([ref_tokens], gen_tokens)
            bleu_scores.append(bleu)
        return bleu_scores
    
    def _get_reference_tokens(self, tokens, q_length):
        offset = self.model.module.prefix_length if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.prefix_length
        return tokens[q_length + offset + 1:].tolist()

    
    """
    Evaluation Metrics
    """
    def eval_task(self, mode):
        print(f"Evaluate on {mode} set:")
        if mode == "val":
            self.args.like_test = True
            self.dev_dataset = self._build_dataset("val")
            dataset = self.dev_dataset
            dataloader = self.val_loader
        elif mode == "train":
            self.args.like_test = True
            self.dev_dataset = self._build_dataset("train")
            dataset = self.train_dataset
            dataloader = self.train_loader
        else:
            dataset = self.test_dataset
            dataloader = self.test_loader

        self.model.eval()
        self.model.to(self.device)

        if self.args.task == "classification":
            self._eval_closed(dataloader=dataloader)
        if self.args.task == "generative":
            self._eval_oa(dataset=dataset)

    def _eval_oa(self, dataset, print_vis_token_meaning=False):
        bert_score = load("bertscore")
        metrics = {
            "bleu_avg1": 0.,
            "bert_avg1": 0.,
            "bert_avg2": 0.,
            "bert_avg3": 0.,
            "f1_avg": 0.,
            "f1_avg_addSpecialTokensFalse": 0.,
            "acc": 0.,
            "acc_oe": 0.,
            "acc_yn": 0.,
            "c_oe": 1e-9,
            "c_yn": 1e-9
        }
        for item in range(len(dataset)):
            metrics = self._process_item(item, metrics, print_vis_token_meaning, dataset, bert_score)
        self._print_metrics(metrics, len(dataset))

    def _eval_closed(self, dataloader):
        self.model.eval()
        self.model.to(self.device)
        all_labels = []
        all_predictions = []
        total_loss = 0.0

        for i, (prefix, tokens, mask, q_len, labels) in enumerate(dataloader): 
            torch.cuda.empty_cache()
            prefix = prefix.to(self.device, dtype=torch.float32)
            tokens = tokens.to(self.device, dtype=torch.long)
            mask = mask.to(self.device, dtype=torch.long)
            q_len = q_len.to(self.device, dtype=torch.long)

            with torch.no_grad():
                outputs = self.model(prefix, tokens, mask, q_len)
                logits = outputs
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=-1)
                loss = self._compute_loss(logits, tokens, q_len, labels)
                total_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        desc = f"Eval Epoch: {self.start_epoch} Loss: {total_loss / (i + 1):.4f} Accuracy: {accuracy:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}"
        print(desc)
        avg_val_loss = total_loss / len(dataloader)
        return avg_val_loss, accuracy, precision, recall, f1

    def _update_metrics(self, answer, out_text, metrics, bert_score):
        out_text_lower = out_text.lower()
        metrics["acc"] += out_text_lower == answer
        if answer in {'yes', 'no'}:
            metrics["acc_yn"] += out_text_lower == answer
            metrics["c_yn"] += 1
        else:
            metrics["acc_oe"] += out_text_lower == answer
            metrics["c_oe"] += 1
        print(f"Answer: {answer}")
        print(f"Output: {out_text_lower}")
        print()

        reference, candidate = [answer], [out_text]
        metrics["bleu_avg1"] += sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        bert_scores = bert_score.compute(references=reference, predictions=candidate, model_type='bert-base-uncased')
        metrics["bert_avg1"] += bert_scores['precision'][0]
        metrics["bert_avg2"] += bert_scores['recall'][0]
        metrics["bert_avg3"] += bert_scores['f1'][0]
        metrics["f1_avg"] += self._compute_f1(self.model.tokenizer.encode(reference[0]), self.model.tokenizer.encode(candidate[0]))
        metrics["f1_avg_addSpecialTokensFalse"] += self._compute_f1(self.model.tokenizer.encode(reference[0], add_special_tokens=False), self.model.tokenizer.encode(candidate[0], add_special_tokens=False))

        return metrics

    def _print_metrics(self, metrics, dataset_size):
        print('------------')
        for key, value in metrics.items():
            print(f"{key}: {round(value / dataset_size, 3)}")

        print('------------')
        print("BLEU {}".format(round(metrics["bleu_avg1"] / dataset_size, 3)))
        print("BERTScore {}".format(round(metrics["bert_avg3"] / dataset_size, 3)))
        print("F1 {}".format(round(metrics["f1_avg"] / dataset_size, 3)))
        print("Accuracy {}".format(round(metrics["acc"] / dataset_size, 3)))
        print("Accuracy YN{}".format(round(metrics["acc_yn"] / metrics["c_yn"], 3)))
        print("Accuracy OE{}".format(round(metrics["acc_oe"] / metrics["c_oe"], 3)))

    def _print_nearest_text_token(self, vis_token):
        """print the nearest token in the vocabulary to the given token through model.gpt.embeddings.weight"""
        if self.args.model_type in ["gpt2","gpt2-xl","microsoft/biogpt"]:
            embeddings = self.model.peftmodel.transformer.wte.weight
        elif self.args.model_type in ["llama","llama2","tiny-llama"]:
            embeddings = self.model.peftmodel.model.model.embed_tokens.weight
        else:
            raise ValueError(f"Unsupported model type: {self.args.model_type}")
        
        distances = torch.norm(embeddings - vis_token, dim=1)
        nearest_token_idx = torch.argmin(distances)
        print(f"NTToken: {self.model.tokenizer.decode([nearest_token_idx.item()])}")

    def _compute_f1(self, gold_toks, pred_toks):
        # Manually calculate the common tokens
        common_tokens = set(gold_toks) & set(pred_toks)
        num_same = sum(min(gold_toks.count(token), pred_toks.count(token)) for token in common_tokens)

        # Return 0 if either list is empty or no common tokens
        if len(gold_toks) == 0 or len(pred_toks) == 0 or num_same == 0:
            return 0.0

        # Calculate precision and recall
        precision = num_same / len(pred_toks)
        recall = num_same / len(gold_toks)

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1


    def _process_item(self, item, metrics, print_vis_token_meaning, dataset, bert_score):
        prefix, tokens, mask, q_len, _ = dataset[item]
        prefix = prefix.type(torch.float32).to(self.device)
        tokens = tokens.type(torch.long).to(self.device)
        mask = mask.type(torch.long).to(self.device)

        print(f"Tokens: {self.model.tokenizer.decode(tokens)}")
        with torch.no_grad():
            embed = self.model.generate_embedding(prefix, tokens, mask, q_len)
            out_text = self._generate_beam(generated=embed, entry_length=dataset.max_seqs_len[1], temperature=1, tokenizer=dataset.tokenizer)[0].lower().strip()

        answer =  str(dataset.answers[item]).lower().strip()
        metrics = self._update_metrics(answer, out_text, metrics, bert_score)
        return metrics

    def _generate_beam(
        self,
        tokenizer,
        beam_size: int = 5,
        generated=None,
        entry_length=65,
        temperature=1.0,
    ):
        self.model.eval()
        stop_token = tokenizer.eos_token
        # encode returns Returns List[int], torch.Tensor, tf.Tensor or np.ndarray
        stop_token_index = self.model.tokenizer.encode(stop_token)[0] #IndexError: list index out of range
        tokens = None
        scores = None
        device = self.device
        seq_lengths = torch.ones(beam_size, device=device)
        is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
        with torch.no_grad():
            for i in range(entry_length):
                outputs = self.model.peftmodel(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()

                if scores is None:
                    # First step initialization
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    # Subsequent steps
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                    
                    # Update beams
                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                
                # Prepare for the next iteration
                next_token_embed = self._create_embeddings(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                if is_stopped.all():
                    break
        # Decode and prepare output        
        scores = scores / seq_lengths
        output_list = tokens.cpu().numpy()
        output_texts = [
            self.model.tokenizer.decode(output[: int(length)])
            for output, length in zip(output_list, seq_lengths)
        ]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i].split(stop_token)[0] for i in order]
        return output_texts