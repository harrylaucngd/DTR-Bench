import os.path
from math import sqrt
import torch
import torch.nn as nn
import gc

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, LlamaForCausalLM,\
    GPT2Config, GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from GlucoseLLM.layers.Embed import PatchEmbedding
import transformers
from GlucoseLLM.layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()

model_hf = {
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "llama-13b": "huggyllama/llama-13b",
    "llama-3-8b": "meta-llama/Llama-3-8b",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-7b": "huggyllama/llama-7b",
    "gpt2": "openaicommunity/gpt2"
}


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, need_llm=True):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.need_llm = need_llm
        self.llm = configs.llm_model

        model_dir = os.path.join("model_hub")
        os.makedirs(model_dir, exist_ok=True)
        if self.need_llm:
            if 'gpt' in configs.llm_model:
                self.llm_config = GPT2Config.from_pretrained(f'{model_dir}/{configs.llm_model}')
                self.llm_config.num_hidden_layers = configs.llm_layers
                self.llm_config.output_attentions = True
                self.llm_config.output_hidden_states = True

                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        f'{model_dir}/{configs.llm_model}',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llm_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        f'{model_hf[configs.llm_model]}',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.llm_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        f'{model_dir}/{configs.llm_model}',
                        cache_dir=f'{model_dir}/{configs.llm_model}',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        f'{model_hf[configs.llm_model]}',
                        cache_dir=f'{model_dir}/{configs.llm_model}',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif 'llama-3' in configs.llm_model:
                self.llm_config = LlamaConfig.from_pretrained(f'{model_dir}/{configs.llm_model}')
                self.llm_config.num_hidden_layers = configs.llm_layers
                self.llm_config.output_attentions = True
                self.llm_config.output_hidden_states = True

                try:
                    self.llm_model = LlamaForCausalLM.from_pretrained(
                        f'{model_dir}/{configs.llm_model}',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llm_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = LlamaForCausalLM.from_pretrained(
                        f'{model_hf[configs.llm_model]}',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.llm_config,
                    )
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        f'{model_dir}/{configs.llm_model}',
                        cache_dir=f'{model_dir}/{configs.llm_model}',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        f'{model_hf[configs.llm_model]}',
                        cache_dir=f'{model_dir}/{configs.llm_model}',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif 'llama' in configs.llm_model:
                self.llm_config = LlamaConfig.from_pretrained(f'{model_dir}/{configs.llm_model}')
                self.llm_config.num_hidden_layers = configs.llm_layers
                self.llm_config.output_attentions = True
                self.llm_config.output_hidden_states = True

                try:
                    self.llm_model = LlamaForCausalLM.from_pretrained(
                        f'{model_dir}/{configs.llm_model}',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llm_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = LlamaForCausalLM.from_pretrained(
                        f'{model_hf[configs.llm_model]}',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.llm_config,
                    )

                try:
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        f'{model_dir}/{configs.llm_model}',
                        cache_dir=f'{model_dir}/{configs.llm_model}',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        f'{model_hf[configs.llm_model]}',
                        cache_dir=f'{model_dir}/{configs.llm_model}',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            else:
                raise ValueError("Unsupported LLM!")
            
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

            for param in self.llm_model.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, prompt):
        dec_out = self.forecast(x_enc, prompt)
        return dec_out[:, -self.pred_len:, :]

    def forecast(self, x_enc, prompt):
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model.model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


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
