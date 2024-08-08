import os.path
from math import sqrt
from typing import List, Dict
from GlucoseLLM.prompts import (SYSTEM_PROMPT, ACTOR_INSTRUCTION_PROMPT, SUMMARY_INSTRUCTION_PROMPT,
                                LLM_INFERENCE_INSTRUCTION_PROMPT, get_Q_instruction, get_patient_info_prompt)
from transformers import AutoModelForCausalLM, AutoTokenizer
from GlucoseLLM.model.Embed import PatchEmbedding
import transformers
import numpy as np

transformers.logging.set_verbosity_error()

model_hf = {
    "internlm2_5-7b-chat": "internlm/internlm2_5-7b-chat",
    "Phi-3-small-128k-instruct": "microsoft/Phi-3-small-128k-instruct",
    "Yi-1.5-9B-Chat": "01-ai/Yi-1.5-9B-Chat",
    "Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen2-7B-Instruct": "Qwen/Qwen2-7B-Instruct",
    "Qwen2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    "Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
}

llm_context_window = {
    "internlm2_5-7b-chat": 32768,
    "Phi-3-small-128k-instruct": 131072,
    "Yi-1.5-9B-Chat": 4096,
    "Meta-Llama-3.1-8B-Instruct": 131072,
    "Qwen2-7B-Instruct": 32768,
    "Qwen2-1.5B-Instruct": 32768,
    "Qwen2-0.5B-Instruct": 32768,
}

import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class FlattenHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class LLMInference(torch.nn.Module):
    def __init__(self, llm="Qwen2-0.5B-Instruct", context_window=896,
                 device="cuda" if torch.cuda.is_available() else "cpu", model_dir=None):
        super().__init__()
        self.llm = llm
        self.max_length = context_window
        self.device = device
        model_dir = "model_hub" if model_dir is None else model_dir
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(f'{model_dir}/{self.llm}',
                                                           trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(f'{model_dir}/{self.llm}',
                                                              trust_remote_code=True).to(self.device)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(f'{model_hf[self.llm]}',
                                                           cache_dir=f'{model_dir}/{self.llm}',
                                                           trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(f'{model_hf[self.llm]}',
                                                              cache_dir=f'{model_dir}/{self.llm}',
                                                              trust_remote_code=True).to(self.device)

    def forward(self, messages: List[Dict]):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=self.max_length,  #todo: change to max new tokens
                do_sample=True,
                temperature=1,  # todo
                top_k=50,  # todo

            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        cutoff_index = generated_text.rfind("assistant\n")
        if cutoff_index != -1:  # answer cutoff
            generated_text = generated_text[cutoff_index + len("assistant\n"):]
        return generated_text


class catLLM(nn.Module):
    pass


class timeLLM(nn.Module):
    def __init__(self, llm, n_vars, output_dim,
                 seq_len,
                 d_model, d_ff, patch_len, stride, token_dim, n_heads, decoder_len, max_new_tokens=512,
                 keep_old=False, dropout: float = 0., model_dir=None,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        :param llm: the name of the LLM model
        :param seq_len: the length of the input sequence
        :param d_model: the dimension of the PatchEmbedding output
        :param d_ff: the dimension of the feed forward layer
        :param patch_len: the length of each patch
        :param stride: the stride of the patch
        :param token_dim: the dimension of the token embedding for LLM
        :param n_heads: the number of heads in the reprogramming layer
        :param enc_in: the number of input features to the encoder
        """
        super(timeLLM, self).__init__()
        self.llm = llm
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.d_ff = d_ff
        self.top_k = 5
        self.d_model = d_model
        self.d_llm = token_dim
        self.patch_len = patch_len
        self.stride = stride
        self.dropout = dropout
        self.n_heads = n_heads
        self.keep_old = keep_old
        self.n_vars = n_vars
        self.n_prototypes = 500
        self.patch_nums = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.decoder_len = decoder_len
        self.head_nf = self.d_ff * self.decoder_len

        self.max_new_tokens = max_new_tokens
        self.device = device
        # find the LLM model and tokenizer
        model_dir = "model_hub" if model_dir is None else model_dir
        os.makedirs(model_dir, exist_ok=True)
        try:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                f'{model_dir}/{self.llm}',
                trust_remote_code=True,
                local_files_only=True,
            )
        except EnvironmentError:  # downloads model from HF if not already done
            print("Local model files not found. Attempting to download...")
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                f'{model_hf[self.llm]}',
                trust_remote_code=True,
                local_files_only=False,
                device_map="auto",
            )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                f'{model_dir}/{self.llm}',
                cache_dir=f'{model_dir}/{self.llm}',
                trust_remote_code=True,
                local_files_only=True,
                device_map="auto",
            )
        except EnvironmentError:  # downloads the tokenizer from HF if not already done
            print("Local tokenizer files not found. Attempting to download them...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                f'{model_hf[self.llm]}',
                cache_dir=f'{model_dir}/{self.llm}',
                trust_remote_code=True,
                local_files_only=False,
                device_map="auto",
            )

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
        self.freeze_llm_model()

        # define reprogramming model
        self.patch_embedding = PatchEmbedding(self.d_model, self.n_vars, self.patch_len, self.stride, self.dropout)

        self.word_embeddings = self.llm_model.model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]

        self.mapping_layer = nn.Linear(self.vocab_size, self.n_prototypes)
        self.reprogramming_layer = ReprogrammingLayer(self.d_model, self.n_heads, self.d_ff, self.d_llm)
        self.output_projection = FlattenHead(self.head_nf, self.output_dim)

        # define old reprogramming model
        if keep_old:
            self.patch_embedding_old = PatchEmbedding(self.d_model, self.n_vars, self.patch_len, self.stride,
                                                      self.dropout)
            self.mapping_layer_old = nn.Linear(self.vocab_size, self.n_prototypes)
            self.reprogramming_layer_old = ReprogrammingLayer(self.d_model, self.n_heads, self.d_ff, self.d_llm)
            self.output_projection = FlattenHead(self.head_nf, self.output_dim)

    def freeze_llm_model(self):
        """Ensure all llm_model parameters are frozen."""
        for param in self.llm_model.parameters():
            param.requires_grad = False

    def unfreeze_llm_model(self):
        """Unfreeze all llm_model parameters, allowing them to be updated during training."""
        for param in self.llm_model.parameters():
            param.requires_grad = True

    def forward(self, x_enc: torch.Tensor, prompt: List[List[Dict]], model="model", state=None, info={}):
        # decide which model to use, current or old
        if model == "model":
            mapping_layer = self.mapping_layer
            patch_embedding = self.patch_embedding
            reprogramming_layer = self.reprogramming_layer
            output_projection = self.output_projection

        elif model == "model_old":
            assert self.keep_old, "Old model is not initialised!"
            mapping_layer = self.mapping_layer_old
            patch_embedding = self.patch_embedding_old
            reprogramming_layer = self.reprogramming_layer_old
            output_projection = self.output_projection_old
        else:
            raise ValueError("Not supported model!")

        # tokenization and embedding
        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                max_length=llm_context_window[self.llm]).input_ids
        prompt_embeddings = self.llm_model.model.get_input_embeddings()(
            prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        # reprogramming time series
        x_enc = torch.tensor(x_enc, dtype=torch.float32).to(self.device)
        source_embeddings = mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = patch_embedding(x_enc)
        enc_out = reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        # llama_enc_out = torch.cat([torch.cat([prompt_embeddings, enc_out[0, :, :].unsqueeze(0)], dim=1), enc_out[1, :, :].unsqueeze(0)], dim=1)
        dec_out = self.llm_model.model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, -self.decoder_len:, :self.d_ff]
        dec_out = output_projection(dec_out)
        return dec_out, None

    def generate_text(self, x_enc: np.array, prompt: List[List[Dict]]):
        """
        Generate text using the LLM model. We allow time series data as prefix.
        """
        # convert conversation into a list of string
        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                max_length=llm_context_window[self.llm]).input_ids
        # Tokenization and embedding

        prompt_embeddings = self.llm_model.model.get_input_embeddings()(
            prompt.to(x_enc.device if x_enc is not None else 'cpu'))  # (batch, prompt_token, dim)

        if x_enc is None:
            llama_enc_out = prompt_embeddings
        else:
            # Reprogramming time series
            source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
            x_enc = x_enc.permute(0, 2, 1).contiguous()
            enc_out, _ = self.patch_embedding(x_enc)
            enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
            llama_enc_out = torch.cat([enc_out, prompt_embeddings], dim=1)

        # Generate text using LLM
        outputs = self.llm_model.generate(
            inputs_embeds=llama_enc_out,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=1
        )

        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        cutoff_index = generated_text.rfind("assistant\n")
        if cutoff_index != -1:  # answer cutoff
            generated_text = generated_text[cutoff_index + len("assistant\n"):]

        return generated_text
