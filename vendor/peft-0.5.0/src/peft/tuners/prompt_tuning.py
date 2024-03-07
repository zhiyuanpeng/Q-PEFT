# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import math
from dataclasses import dataclass, field
from typing import Optional, Union
import pickle
import torch
import torch.nn as nn
from ..config import PromptLearningConfig
from ..utils import PeftType


class PromptTuningInit(str, enum.Enum):
    TEXT = "TEXT"
    RANDOM = "RANDOM"


@dataclass
class PromptTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PromptEmbedding`].

    Args:
        prompt_tuning_init (Union[[`PromptTuningInit`], `str`]): The initialization of the prompt embedding.
        prompt_tuning_init_text (`str`, *optional*):
            The text to initialize the prompt embedding. Only used if `prompt_tuning_init` is `TEXT`.
        tokenizer_name_or_path (`str`, *optional*):
            The name or path of the tokenizer. Only used if `prompt_tuning_init` is `TEXT`.
    """
    update_doc_embd: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Set true to update the embedding layer of second doc (the one after docspec doc)"
        },
    )
    spt_layer_num: Optional[int] = field(
        default=0,
        metadata={
            "help": "Set the number of liner layers after the soft prompt's embedding layer"
        },
    )

    docspec_layer_num: Optional[int] = field(
        default=0,
        metadata={
            "help": "Set the number of liner layers after the doc spec's embedding layer"
        },
    )

    fp16train: Optional[bool] = field(
        default=False,
        metadata={
            "help": "true to enable fp16train"
        },
    )

    docspec: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Set True to enable doc specific"
        },
    )
    spt: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Set True to enable soft prompt tuning"
        },
    )

    fix_docspec_embed: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Set True to fix the docspec embedding matrix"
        },
    )

    exp_k: Optional[int] = field(
        default=1,
        metadata={
            "help": "# of expanded tokens"
        },
    )

    exp_mode: Optional[int] = field(
        default=1,
        metadata={
            "help": "set 1: for each token, retrieve exp_k tokens; set 2: for each query expand to top exp_k tokens"
        },
    )

    exp_head_num: Optional[int] = field(
        default=0,
        metadata={
            "help": "for mode 4, attention haed num"
        },
    )

    exp_loc: Optional[str] = field(
        default="suffix",
        metadata={
            "help": "append expanded tokens to prefix or suffix the document"
        },
    )

    exp_normalize: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Set True to apply normalize on q_v matrix"
        },
    )

    exp_random: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Set True to select random k tokens from V for mode 2 and from document for mode 3"
        },
    )

    exp_filter_dict_path: Optional[str] = field(
        default="./token_filter.pkl",
        metadata={
            "help": "blacklist for tokens to be filtered out"
        },
    )

    exp_filter_mode: Optional[str] = field(
        default="n",
        metadata={
            "help": "n: None, d: duplicate, ds: duplicate and stopword, da: duplicate and alpha, dsa: duplicate, stopword and alpha"
        },
    )

    exp_prompt: Optional[str] = field(
        default="hints",
        metadata={
            "help": "prompt before the retrieved topk tokens"
        },
    )

    prompt_tuning_init: Union[PromptTuningInit, str] = field(
        default=PromptTuningInit.RANDOM,
        metadata={"help": "How to initialize the prompt tuning parameters"},
    )
    prompt_tuning_init_text: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.PROMPT_TUNING


class PromptEmbedding(torch.nn.Module):
    """
    The model to encode virtual tokens into prompt embeddings.

    Args:
        config ([`PromptTuningConfig`]): The configuration of the prompt embedding.
        word_embeddings (`torch.nn.Module`): The word embeddings of the base transformer model.

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prompt embedding.

    Example:

    ```py
    >>> from peft import PromptEmbedding, PromptTuningConfig

    >>> config = PromptTuningConfig(
    ...     peft_type="PROMPT_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     prompt_tuning_init="TEXT",
    ...     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
    ...     tokenizer_name_or_path="t5-base",
    ... )

    >>> # t5_model.shared is the word embeddings of the base model
    >>> prompt_embedding = PromptEmbedding(config, t5_model.shared)
    ```

    Input Shape: (`batch_size`, `total_virtual_tokens`)

    Output Shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """

    def __init__(self, config, word_embeddings):
        super().__init__()
        #word_embeddings will be bf16 is load llam bf16 otherwise float32
        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules # *1
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim)
        if config.prompt_tuning_init == PromptTuningInit.TEXT:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
            init_text = config.prompt_tuning_init_text
            init_token_ids = self.tokenizer(init_text)["input_ids"][1:]
            # Trim or iterate until num_text_tokens matches total_virtual_tokens
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > total_virtual_tokens:
                init_token_ids = init_token_ids[:total_virtual_tokens]
            elif num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_virtual_tokens]
            # len(init_token_ids) = total_virtual_tokens
            word_embedding_weights = word_embeddings(torch.LongTensor(init_token_ids)).detach().clone() # word_embedding_weights.shape = total_virtual_tokens*dim (50*4096)
            # word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights, requires_grad=config.spt) # init self.embed
            print(f"soft prompt tuning is {config.spt}")
            # self defined multi layers after embedding
            embd_list = []
            if not config.spt:
                assert config.spt_layer_num == 0, "must set spt_layer_num=0, if disable spt"
            # MLP begins with a linear layer
            if config.spt_layer_num > 0:
                if config.fp16train:
                    embd_list.append(torch.nn.Linear(config.token_dim, config.token_dim).to(torch.bfloat16))
                else:
                    embd_list.append(torch.nn.Linear(config.token_dim, config.token_dim))
            for _ in range(config.spt_layer_num):
                embd_list.append(torch.nn.ReLU())
                if config.fp16train:
                    embd_list.append(torch.nn.Linear(config.token_dim, config.token_dim).to(torch.bfloat16))
                else:
                    embd_list.append(torch.nn.Linear(config.token_dim, config.token_dim))
            if embd_list:# >=1 layer, last is relu
                self.spt_multi_layers = torch.nn.Sequential(*embd_list)
            else:
                self.spt_multi_layers = None
        print(f"doc specific is {config.docspec}")
        if config.docspec:
            # intialized docspec_word_embeddings with word_embeddings
            self.docspec_embeddings = torch.nn.Embedding(word_embeddings.weight.shape[0], word_embeddings.weight.shape[1])
            self.docspec_embeddings.weight = torch.nn.Parameter(word_embeddings.weight.detach().clone(), requires_grad= not config.fix_docspec_embed)
            if config.fix_docspec_embed:
                assert config.docspec_layer_num == 0, "set docspec_layer_num=0, if fix docspec"
            docspec_embd_list = []
            # MLP begins with a linear layer
            if config.docspec_layer_num > 0:
                if config.fp16train:
                    docspec_embd_list.append(torch.nn.Linear(config.token_dim, config.token_dim).to(torch.bfloat16))
                else:
                    docspec_embd_list.append(torch.nn.Linear(config.token_dim, config.token_dim))
            for _ in range(config.docspec_layer_num):
                docspec_embd_list.append(torch.nn.ReLU())
                if config.fp16train:
                    docspec_embd_list.append(torch.nn.Linear(config.token_dim, config.token_dim).to(torch.bfloat16))
                else:
                    docspec_embd_list.append(torch.nn.Linear(config.token_dim, config.token_dim))
            if docspec_embd_list: # >=1 layer, last is relu
                self.doc_multi_layers = torch.nn.Sequential(*docspec_embd_list)
            else:
                self.doc_multi_layers = None
            if config.exp_mode == 4:
                if config.fp16train:
                    self.docspec_attention = nn.MultiheadAttention(config.token_dim, num_heads=config.exp_head_num, dropout=0.1, batch_first=True).to(torch.bfloat16)
                else:
                    self.docspec_attention = nn.MultiheadAttention(config.token_dim, num_heads=config.exp_head_num, dropout=0.1, batch_first=True)
        self.exp_k = config.exp_k
        self.exp_mode = config.exp_mode
        self.exp_head_num = config.exp_head_num
        self.exp_normalize = config.exp_normalize
        self.exp_random = config.exp_random
        self.exp_filter_mode = config.exp_filter_mode
        if config.exp_filter_dict_path is not None:
            self.exp_filter_dict = pickle.load(open(config.exp_filter_dict_path, 'rb'))
            self.blacklist_ids_s = set(self.exp_filter_dict["stopword_keys"])
            self.blacklist_ids_a = set(self.exp_filter_dict["non_alphabetic_keys"])
        else:
            self.exp_filter_dict = None
        self.hints = self.tokenizer(f"{config.exp_prompt}: ")
        self.hints_end = self.tokenizer("\n")
        self.start = self.tokenizer("<s>")
        self.start_embed = word_embeddings(torch.LongTensor(self.start["input_ids"][1:])).detach().clone()

    def filter_ids(self, d_k_ids, blacklist_ids):
        filtered_tensors = []
        for tensor in d_k_ids:
            mask = torch.tensor([item not in blacklist_ids for item in tensor.tolist()], dtype=torch.bool)
            filtered_tensors.append(tensor[mask])
        return filtered_tensors
    
    def filter_eq(self, d_k_ids, docspec_ids, q_len, device):
        # "[None], [duplicate], [duplicate, stopwords], [duplicate, alphabetic], [duplicate, stopwords, alphabetic]"
        ori_d_k_ids = d_k_ids
        # by default, filter out the retrieved tokens in the query
        if self.exp_mode == 2 and "q" in self.exp_filter_mode:
            d_k_ids = d_k_ids[:, q_len:]
        if "d" in self.exp_filter_mode:
            # print("filter duplicate tokens")
            unique_tensors = []
            for row in d_k_ids:
                unique_elements = torch.unique(row, sorted=False)
                indices = torch.cat([torch.where(row == u)[0][:1] for u in unique_elements]).sort().values
                unique_row_in_original_order = row[indices]
                unique_tensors.append(unique_row_in_original_order)
            d_k_ids = unique_tensors
        if self.exp_mode == 3 and "q" in self.exp_filter_mode:
            # for tok q_len tokens, if the token is in docspec_ids(query ids), then remove it
            unique_tensors = []
            for row_id, (d_k_ids_row, docspec_ids_row) in enumerate(zip(d_k_ids, docspec_ids)):
                unique_tensor = []
                for i in range(len(d_k_ids_row)):
                    if d_k_ids_row[i] in docspec_ids_row:
                        continue
                    else:
                        unique_tensor.append(d_k_ids_row[i])
                unique_tensors.append(torch.tensor(unique_tensor))
            d_k_ids = unique_tensors
        if "s" in self.exp_filter_mode:
            # print("filter stopwords")
            d_k_ids = self.filter_ids(d_k_ids, self.blacklist_ids_s)
        if "a" in self.exp_filter_mode:
            # print("filter alphabetic tokens")
            d_k_ids = self.filter_ids(d_k_ids, self.blacklist_ids_a) 
        # for mode 2, its less likely to have 1,2 tokens in d_k_ids
        d_k_ids = self.filter_ids(d_k_ids, [1,2])
        ans = []
        for tensor, ori_tensor in zip(d_k_ids, ori_d_k_ids):
            tensor = tensor.to(device)
            if len(tensor) < self.exp_k:
                if len(tensor) == 0:
                    print("no token left after filtering, do not apply filter for this query")
                    tensor = ori_tensor
                else:
                    print("not enough tokens to expand, oversample to exp_k tokens")
                    indices = torch.randint(len(tensor), (self.exp_k,))
                    tensor = tensor[indices]
            ans.append(tensor[:self.exp_k])
        d_k_ids = torch.stack(ans)
        return d_k_ids.to(device)
    
    def mode4filter(self, doc_indices, input_ids, input_masks, device):
        # filter out the tokens in document
        filtered_input_masks = input_masks.detach().clone()
        # by default, mask 1 and 2
        for (input_row, mask_row, q_row) in zip(input_ids, filtered_input_masks, doc_indices):
            for i in range(len(input_row)):
                # do not filter already masked tokens
                if mask_row[i] != 0:
                    if input_row[i] in [1,2]:
                        mask_row[i] = 0
                    if "q" in self.exp_filter_mode:
                        if input_row[i] in q_row:
                            mask_row[i] = 0
                    if "s" in self.exp_filter_mode:
                        # print("filter stopwords")
                        if int(input_row[i]) in self.blacklist_ids_s:
                            mask_row[i] = 0
                    if "a" in self.exp_filter_mode:
                        # print("filter alphabetic tokens")
                        if int(input_row[i]) in self.blacklist_ids_a:
                            mask_row[i] = 0
        return filtered_input_masks.to(device)

    def mask_query_in_input(self, input_ids, input_masks, device):
        # mask out the query itself, input_ids=padding + doc + 1 + query + 2
        only_doc_mask = input_masks.detach().clone()
        for (input_row, mask_row) in zip(input_ids, only_doc_mask):
            for i in range(len(input_row)-1, -1, -1):
                # do not filter already masked tokens
                if input_row[i] == 1:
                    mask_row[i] = 0
                    break
                else:
                    mask_row[i] = 0
        return only_doc_mask.to(device)

    def random_select_k(self, input_ids, input_masks, k, device):
        # Find the indices of the unmasked tokens for each row
        unmasked_indices = torch.where(input_masks == 1)

        # Initialize a tensor to hold the selected ids
        selected_ids = torch.zeros((input_ids.shape[0], k), dtype=torch.long)

        # For each row in the input
        for i in range(input_ids.shape[0]):
            # Get the unmasked indices for this row
            indices = unmasked_indices[1][unmasked_indices[0] == i]

            # Randomly select k of these indices
            if indices.shape[0] < k:
                # print("Random Experiments: k is too big, longer than the number of unmasked tokens")
                selected = torch.randint(0, indices.shape[0], (k,))
            else:
                selected = torch.randperm(indices.shape[0])[:k]

            # Get the ids corresponding to these indices
            ids = input_ids[i, indices[selected]]

            # Add these ids to the selected_ids tensor
            selected_ids[i] = ids

        return selected_ids.to(device)
    
    def query_expansion(self, doc_indices, doc_attention_mask, input_ids, input_embeds, input_masks, device):
        batch_size = doc_indices.size(0)
        docspec_embeddings = self.docspec_embeddings(doc_indices) #batch_size*doc_len*dim
        if self.exp_random:
            if self.exp_mode == 3:
                # random select k from query
                d_k_ids = self.random_select_k(input_ids, input_masks, self.exp_k, device)
            if self.exp_mode == 2:
                # random select k from input_embeds
                d_k_ids = self.random_select_k(doc_indices, doc_attention_mask, self.exp_k, device)
            d_k_ids = torch.cat([torch.tensor(self.hints["input_ids"][1:]).to(device).unsqueeze(0).expand(batch_size, -1), d_k_ids, torch.tensor(self.hints_end["input_ids"][1:]).to(device).unsqueeze(0).expand(batch_size, -1)], dim=1)
            docspec_embeddings = self.docspec_embeddings(d_k_ids)
            docspec_attention_mask = torch.ones_like(d_k_ids)
            docspec_labels = torch.full_like(d_k_ids, -100)
            return docspec_embeddings, docspec_attention_mask, docspec_labels, d_k_ids
        if self.exp_mode == 4:
            # attention mode, query attention on document
            # by default, mask the query itself in input
            filtered_input_masks = self.mask_query_in_input(input_ids, input_masks, device)
            filtered_input_masks = self.mode4filter(doc_indices, input_ids, filtered_input_masks, device)
            docspec_embeddings, _ = self.docspec_attention(docspec_embeddings, input_embeds, input_embeds, key_padding_mask=~filtered_input_masks.to(torch.bool), attn_mask=None)
            hints_embed = self.docspec_embeddings(torch.tensor(self.hints["input_ids"][1:]).to(device))
            hints_end_embed = self.docspec_embeddings(torch.tensor(self.hints_end["input_ids"][1:]).to(device))
            docspec_embeddings = torch.cat([hints_embed.unsqueeze(0).expand(docspec_embeddings.size(0), -1, -1), docspec_embeddings, hints_end_embed.unsqueeze(0).expand(docspec_embeddings.size(0), -1, -1)], dim=1)
            docspec_attention_mask = torch.cat([torch.ones((doc_attention_mask.size(0), hints_embed.size(0))).to(device), doc_attention_mask, torch.ones((doc_attention_mask.size(0), hints_end_embed.size(0))).to(device)], dim=1)
            docspec_labels = torch.full_like(docspec_attention_mask, -100).to(device)
            d_k_ids = torch.cat([torch.tensor(self.hints["input_ids"][1:]).to(device).unsqueeze(0).expand(batch_size, -1), doc_indices, torch.tensor(self.hints_end["input_ids"][1:]).to(device).unsqueeze(0).expand(batch_size, -1)], dim=1)
            return docspec_embeddings, docspec_attention_mask, docspec_labels, d_k_ids
        # exp_mode 1, 2: retrieve top k tokens from V, where V is the embedding matrix. The difference between mode 1 and 2 is that mode 1 retrieve top k tokens for each token, while mode 2 retrieve top k tokens for each query
        if self.exp_mode in [1, 2]:
            v_t = torch.transpose(self.docspec_embeddings.weight, 0, 1)# dim*|V|
            v_t = v_t.repeat(docspec_embeddings.size(0), 1, 1)# batch_size*dim*|V|
            if self.exp_normalize:
                docspec_embeddings_normalize = torch.nn.functional.normalize(docspec_embeddings, dim=-1)
                v_t_softmax_normalize = torch.nn.functional.normalize(v_t, dim=1)
                docspec_embed_dot = torch.bmm(docspec_embeddings_normalize, v_t_softmax_normalize)# batch_size*doc_len*|V| 
            else:
                docspec_embed_dot = torch.bmm(docspec_embeddings, v_t)# batch_size*doc_len*|V|
        # exp_mode 3: instead of retrieving top k tokens from V, retrieve top k tokens from the document
        if self.exp_mode == 3:
            input_embeds = torch.transpose(input_embeds, 1, 2)
            if self.exp_normalize:
                docspec_embeddings_normalize = torch.nn.functional.normalize(docspec_embeddings, dim=-1)
                input_embeds_normalize = torch.nn.functional.normalize(input_embeds, dim=1)
                docspec_embed_dot = torch.bmm(docspec_embeddings_normalize, input_embeds_normalize)# batch_size*doc_len*|V| 
            else:
                docspec_embed_dot = torch.bmm(docspec_embeddings, input_embeds)# 
        # exp_mode 1: for each token, retrieve top k tokens
        if self.exp_mode == 1:
            d_k_scores, d_k_ids = torch.topk(docspec_embed_dot, k=self.exp_k+1, dim=2)# b*doc_len*(exp_k+1)
            # remove the top 1 token, which is the docspec token itself
            d_k_ids = d_k_ids[:, :, 1:]
            # if self.exp_k=2, expand query to query1+query2 instead of q11+q21+q12+q22...
            d_k_ids = torch.transpose(d_k_ids, 1, 2)
            d_k_ids = d_k_ids.reshape(d_k_ids.size(0), -1)
            docspec_embeddings = self.docspec_embeddings(d_k_ids)
            # for attention mask
            docspec_attention_mask = doc_attention_mask.unsqueeze(1).expand(-1,self.exp_k,-1)
            docspec_attention_mask = docspec_attention_mask.reshape(docspec_attention_mask.size(0), -1)
            docspec_labels = torch.full_like(docspec_attention_mask, -100)
            return docspec_embeddings, docspec_attention_mask, docspec_labels, d_k_ids
        # shared by mode 2,3
        docspec_embed_dot = torch.sigmoid(docspec_embed_dot)# map (-1,1) to (0,1)
        docspec_embed_dot = docspec_embed_dot * doc_attention_mask.unsqueeze(-1) # mask out the padding tokens
        if self.exp_mode == 3:
            only_doc_mask = self.mask_query_in_input(input_ids, input_masks, device)
            docspec_embed_dot = docspec_embed_dot * only_doc_mask.unsqueeze(1) # mask out the padding tokens in document, and the query itself
        batch_size, doc_len, token_num = docspec_embed_dot.size()
        if self.exp_mode == 3:
            docspec_embed_dot_ori_index = input_ids.unsqueeze(1).expand(-1, doc_len, -1)
        if self.exp_mode == 2:
            docspec_embed_dot_ori_index = torch.arange(token_num).unsqueeze(0).repeat(batch_size * doc_len, 1).view(batch_size, doc_len, token_num).to(device)
        docspec_embed_dot = docspec_embed_dot.reshape(docspec_embed_dot.size(0), -1)
        docspec_embed_dot_ori_index = docspec_embed_dot_ori_index.reshape(docspec_embed_dot_ori_index.size(0), -1)
        # if filter, then retrieve more 10 times tokens as the number of tokens to be expanded
        d_k_score, d_k_ids = torch.topk(docspec_embed_dot, k=min(1000, docspec_embed_dot.size(1)), dim=1)
        d_k_ids = docspec_embed_dot_ori_index.gather(1, d_k_ids)
        d_k_ids = self.filter_eq(d_k_ids, doc_indices, doc_attention_mask.size(1), device)
        d_k_ids = torch.cat([torch.tensor(self.hints["input_ids"][1:]).to(device).unsqueeze(0).expand(batch_size, -1), d_k_ids, torch.tensor(self.hints_end["input_ids"][1:]).to(device).unsqueeze(0).expand(batch_size, -1)], dim=1)
        docspec_embeddings = self.docspec_embeddings(d_k_ids)
        docspec_attention_mask = torch.ones_like(d_k_ids)
        docspec_labels = torch.full_like(d_k_ids, -100)
        return docspec_embeddings, docspec_attention_mask, docspec_labels, d_k_ids
            
    def forward(self, prompt_indices, doc_indices, doc_attention_mask, input_ids, input_embeds, input_masks):
        device = prompt_indices.device
        # Just get embeddings
        prompt_embeddings = self.embedding(prompt_indices)
        # add multi layer
        if self.spt_multi_layers is not None:
            prompt_embeddings = self.spt_multi_layers(prompt_embeddings)
        # Just get embeddings
        if doc_indices is not None:
            # call the corresponding query expansion method
            docspec_embeddings, docspec_attention_mask, docspec_labels, d_k_ids = self.query_expansion(doc_indices, doc_attention_mask, input_ids, input_embeds, input_masks, device)
            # add multi layer
            if self.doc_multi_layers is not None:
                docspec_embeddings = self.doc_multi_layers(docspec_embeddings)
            return torch.cat([self.start_embed.unsqueeze(1).expand(prompt_embeddings.size(0), -1, -1).to(device), prompt_embeddings.to(device)], dim=1), docspec_embeddings.to(device), docspec_attention_mask.to(device), docspec_labels.to(device), d_k_ids
        else:
            return torch.cat([self.start_embed.unsqueeze(1).expand(prompt_embeddings.size(0), -1, -1).to(device), prompt_embeddings.to(device)], dim=1), None, None, None, None