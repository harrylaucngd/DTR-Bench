import os.path
from math import sqrt
from typing import List, Dict
from GlucoseLLM.prompts import (SYSTEM_PROMPT, ACTOR_INSTRUCTION_PROMPT, SUMMARY_INSTRUCTION_PROMPT,
                                LLM_INFERENCE_INSTRUCTION_PROMPT, get_Q_instruction, get_patient_info_prompt)
from transformers import AutoModelForCausalLM, AutoTokenizer
from GlucoseLLM.model.Embed import PatchEmbedding
import transformers

transformers.logging.set_verbosity_error()

model_hf = {
    "internlm2_5-7b-chat": "internlm/internlm2_5-7b-chat",
    "Phi-3-small-128k-instruct": "microsoft/Phi-3-small-128k-instruct",
    "Yi-1.5-9B-Chat": "01-ai/Yi-1.5-9B-Chat",
    "Qwen2-7B-Instruct": "Qwen/Qwen2-7B-Instruct",
    "Qwen2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    "Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
}

llm_context_window = {
    "internlm2_5-7b-chat": 32768,
    "Phi-3-small-128k-instruct": 131072,
    "Yi-1.5-9B-Chat": 4096,
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
        self.tokenizer = AutoTokenizer.from_pretrained(f'{model_hf[self.llm]}',
                                                       cache_dir=f'{model_dir}/{self.llm}',
                                                       trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(f'{model_hf[self.llm]}',
                                                          cache_dir=f'{model_dir}/{self.llm}',
                                                          trust_remote_code=True).to(self.device)

    def forward(self, messages:List[Dict]):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        cutoff_index = generated_text.rfind("assistant\n")
        if cutoff_index != -1:  # answer cutoff
            generated_text = generated_text[cutoff_index + len("assistant\n"):]
        return generated_text


class catLLM(nn.Module):
    pass


class timeLLM(nn.Module):
    def __init__(self, llm, seq_len, d_model, d_ff, patch_len, stride, token_dim, n_heads, enc_in,
                 keep_old=False, dropout: float = 0., model_dir=None, ):
        super(timeLLM, self).__init__()
        self.llm = llm
        self.seq_len = seq_len
        self.d_ff = d_ff
        self.top_k = 5
        self.d_model = d_model
        self.d_llm = token_dim
        self.patch_len = patch_len
        self.stride = stride
        self.dropout = dropout
        self.n_heads = n_heads
        self.enc_in = enc_in
        self.keep_old = keep_old

        self.num_tokens = 8192
        self.patch_nums = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

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
        self.patch_embedding = PatchEmbedding(self.d_model, self.patch_len, self.stride, self.dropout)

        self.word_embeddings = self.llm_model.model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]

        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(self.d_model, self.n_heads, self.d_ff, self.d_llm)
        self.output_projection = FlattenHead(self.enc_in, self.head_nf)

        # define old reprogramming model
        if keep_old:
            self.patch_embedding_old = PatchEmbedding(self.d_model, self.patch_len, self.stride, self.dropout)
            self.mapping_layer_old = nn.Linear(self.vocab_size, self.num_tokens)
            self.reprogramming_layer_old = ReprogrammingLayer(self.d_model, self.n_heads, self.d_ff, self.d_llm)
            self.output_projection_old = FlattenHead(self.enc_in, self.head_nf)

    def freeze_llm_model(self):
        """Ensure all llm_model parameters are frozen."""
        for param in self.llm_model.parameters():
            param.requires_grad = False

    def unfreeze_llm_model(self):
        """Unfreeze all llm_model parameters, allowing them to be updated during training."""
        for param in self.llm_model.parameters():
            param.requires_grad = True

    def forward(self, x_enc, prompt, model="current", state=None, info={}):
        # decide which model to use, current or old
        if model == "current":
            mapping_layer = self.mapping_layer
            patch_embedding = self.patch_embedding
            reprogramming_layer = self.reprogramming_layer
            output_projection = self.output_projection

        elif model == "old":
            assert self.keep_old, "Old model is not initialised!"
            mapping_layer = self.mapping_layer_old
            patch_embedding = self.patch_embedding_old
            reprogramming_layer = self.reprogramming_layer_old
            output_projection = self.output_projection_old
        else:
            raise ValueError("Not supported model!")

        # tokenization and embedding
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                max_length=llm_context_window[self.llm]).input_ids
        prompt_embeddings = self.llm_model.model.get_input_embeddings()(
            prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        # reprogramming time series
        source_embeddings = mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = patch_embedding(x_enc)
        enc_out = reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        # llama_enc_out = torch.cat([torch.cat([prompt_embeddings, enc_out[0, :, :].unsqueeze(0)], dim=1), enc_out[1, :, :].unsqueeze(0)], dim=1)
        dec_out = self.llm_model.model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        return dec_out[:, -self.pred_len:, :], None

    def generate_text(self, x_enc, prompt, max_length=256):
        # Check the type of the prompt
        if isinstance(prompt, list):
            # If the prompt is a list of dictionaries, convert each conversation into a string format
            prompt = [self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for
                      message in prompt]
            prompt = ' '.join(prompt)
        elif isinstance(prompt, str):
            # If the prompt is a string, use it as it is
            pass
        else:
            raise ValueError("Unsupported prompt type! The prompt should be either a string or a list of dictionaries.")

        # Tokenization and embedding
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                max_length=llm_context_window[self.llm]).input_ids
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
            llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)

        # Generate text using LLM
        outputs = self.llm_model.generate(
            inputs_embeds=llama_enc_out,
            max_length=self.max_length,
            do_sample=True,
            temperature=1
        )

        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        cutoff_index = generated_text.rfind("assistant\n")
        if cutoff_index != -1:  # answer cutoff
            generated_text = generated_text[cutoff_index + len("assistant\n"):]

        return generated_text
