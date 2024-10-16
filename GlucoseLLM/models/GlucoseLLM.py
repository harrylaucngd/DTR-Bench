import os.path
from math import sqrt
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from GlucoseLLM.models.layers.Embed import PatchEmbedding
import transformers
from GlucoseLLM.models.layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()

model_hf = {
    "internlm2_5-7b-chat": "internlm/internlm2_5-7b-chat",
    "Phi-3-small-128k-instruct": "microsoft/Phi-3-small-128k-instruct",
    "Yi-1.5-9B-Chat": "01-ai/Yi-1.5-9B-Chat",
    "Qwen2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    "Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
}

llm_context_window = {
    "internlm2_5-7b-chat": 32768,
    "Phi-3-small-128k-instruct": 131072,
    "Yi-1.5-9B-Chat": 4096,
    "Qwen2-1.5B-Instruct": 32768,
    "Qwen2-0.5B-Instruct": 32768,
}


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0, dtype=torch.float32):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window, dtype=dtype)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, need_llm=True, dtype=torch.float32):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.token_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.need_llm = need_llm
        self.llm_name = configs.llm
        self.dtype = dtype

        model_dir = os.path.join("model_hub")
        os.makedirs(model_dir, exist_ok=True)
        if self.need_llm:
            try:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    f'{model_dir}/{configs.llm}',
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=self.dtype
                )
            except EnvironmentError:  # downloads model from HF if not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    f'{model_hf[configs.llm]}',
                    trust_remote_code=True,
                    local_files_only=False,
                    torch_dtype=self.dtype
                )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    f'{model_dir}/{configs.llm}',
                    cache_dir=f'{model_dir}/{configs.llm}',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Attempting to download them...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    f'{model_hf[configs.llm]}',
                    cache_dir=f'{model_dir}/{configs.llm}',
                    trust_remote_code=True,
                    local_files_only=False
                )

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
            configs.d_model, self.patch_len, self.stride, configs.dropout, dtype=self.dtype)
        self.patch_embedding_old = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout, dtype=self.dtype)

        self.word_embeddings = self.llm_model.model.get_input_embeddings().weight
        self.word_embeddings = self.word_embeddings.to(dtype=self.dtype)

        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens, dtype=self.dtype)
        self.mapping_layer_old = nn.Linear(self.vocab_size, self.num_tokens, dtype=self.dtype)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm, dtype=self.dtype)
        self.reprogramming_layer_old = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm, dtype=self.dtype)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                             head_dropout=configs.dropout, dtype=self.dtype)
        self.output_projection_old = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout, dtype=self.dtype)

        self.active_branch = "model"

        # Convert the entire model to the desired dtype
        self.to(dtype=self.dtype)

    def forward(self, x_enc, prompt):
        x_enc = x_enc.to(dtype=self.dtype)
        dec_out = self.forecast(x_enc, prompt)
        return dec_out[:, -self.pred_len:, :]

    def forecast(self, x_enc, prompt):
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, padding_side="left",
                                max_length=llm_context_window[self.llm_name]).input_ids
        prompt = prompt.to(device=x_enc.device)

        prompt_embeddings = self.llm_model.model.get_input_embeddings()(prompt)
        prompt_embeddings = prompt_embeddings.to(dtype=self.dtype)

        if self.active_branch == "model":
            source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        elif self.active_branch == "model_old":
            source_embeddings = self.mapping_layer_old(self.word_embeddings.permute(1, 0)).permute(1, 0)
        else:
            raise ValueError("Not supported branch!")

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        if self.active_branch == "model":
            enc_out, n_vars = self.patch_embedding(x_enc)
            enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        elif self.active_branch == "model_old":
            enc_out, n_vars = self.patch_embedding_old(x_enc)
            enc_out = self.reprogramming_layer_old(enc_out, source_embeddings, source_embeddings)
        else:
            raise ValueError("Not supported branch!")
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model.model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        if self.active_branch == "model":
            dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        elif self.active_branch == "model_old":
            dec_out = self.output_projection_old(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        return dec_out


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1, dtype=torch.float32):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads, dtype=dtype)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads, dtype=dtype)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads, dtype=dtype)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm, dtype=dtype)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)
        self.dtype = dtype

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
        scores = scores.to(dtype=self.dtype)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
