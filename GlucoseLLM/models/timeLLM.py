import os.path
from math import sqrt

import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

transformers.logging.set_verbosity_error()

# Mapping of model names to their Hugging Face identifiers
model_hf = {
    "internlm2_5-7b-chat": "internlm/internlm2_5-7b-chat",
    "Phi-3-small-128k-instruct": "microsoft/Phi-3-small-128k-instruct",
    "Yi-1.5-9B-Chat": "01-ai/Yi-1.5-9B-Chat",
    "Qwen2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    "Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
}


class FlattenHead(nn.Module):
    """
    A module to flatten the output from the LLM and project it to the target window size.
    """

    def __init__(self, n_vars, nf, target_window, head_dropout=0, dtype=torch.float32):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)  # Flatten the last two dimensions
        self.linear = nn.Linear(nf, target_window, dtype=dtype)  # Project to target window
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        Forward pass for FlattenHead.

        Args:
            x (Tensor): Shape [batch_size, n_vars, d_ff, num_patches]

        Returns:
            Tensor: Shape [batch_size, n_vars, target_window]
        """
        # x shape: [batch_size, n_vars, d_ff, num_patches]
        x = self.flatten(x)
        # x shape after flatten: [batch_size, n_vars, d_ff * num_patches]
        x = self.linear(x)
        # x shape after linear: [batch_size, n_vars, target_window]
        x = self.dropout(x)
        return x


class ReprogrammingLayer(nn.Module):
    """
    A multi-head cross-attention layer to reprogram time series patches using text prototypes.
    """

    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1, dtype=torch.float32):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        # Linear projections for queries, keys, and values
        self.query_projection = nn.Linear(d_model, d_keys * n_heads, dtype=dtype)  # From time series embedding
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads, dtype=dtype)  # From text prototypes
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads, dtype=dtype)  # From text prototypes
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm, dtype=dtype)  # To LLM embedding dimension

        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)
        self.dtype = dtype

    def forward(self, target_embedding, source_embedding, value_embedding):
        """
        Forward pass for ReprogrammingLayer.

        Args:
            target_embedding (Tensor): Shape [batch_size, seq_len, d_model]
            source_embedding (Tensor): Shape [num_prototypes, d_llm]
            value_embedding (Tensor): Shape [num_prototypes, d_llm]

        Returns:
            Tensor: Shape [batch_size, seq_len, d_llm]
        """
        B, L, _ = target_embedding.shape  # Batch size, sequence length, d_model
        S, _ = source_embedding.shape  # Number of prototypes, d_llm
        H = self.n_heads  # Number of attention heads

        # Project queries, keys, and values
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)  # [B, L, H, d_keys]
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)  # [S, H, d_keys]
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)  # [S, H, d_keys]

        # Perform multi-head cross-attention
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)  # [B, L, H, d_keys]

        out = out.reshape(B, L, -1)  # [B, L, H * d_keys]
        return self.out_projection(out)  # [B, L, d_llm]

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        """
        Computes the cross-attention between target and source embeddings.

        Args:
            target_embedding (Tensor): Shape [B, L, H, E]
            source_embedding (Tensor): Shape [S, H, E]
            value_embedding (Tensor): Shape [S, H, E]

        Returns:
            Tensor: Shape [B, L, H, E]
        """
        B, L, H, E = target_embedding.shape  # Batch size, sequence length, num_heads, d_keys

        scale = 1.0 / sqrt(E)

        # Compute attention scores
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)  # [B, H, L, S]
        scores = scores.to(dtype=self.dtype)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B, H, L, S]

        # Compute attention output
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)  # [B, L, H, E]
        return reprogramming_embedding


class TokenEmbedding(nn.Module):
    """
    A token embedding layer using a 1D convolution.
    """

    def __init__(self, c_in, d_model, dtype=torch.float32):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,  # Number of input channels (patch length)
            out_channels=d_model,  # Number of output channels (embedding dimension)
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            dtype=dtype,
        )
        # Initialize convolution weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.dtype = dtype

    def forward(self, x):
        """
        Forward pass for TokenEmbedding.

        Args:
            x (Tensor): Shape [batch_size, c_in, seq_len]

        Returns:
            Tensor: Shape [batch_size, d_model, seq_len]
        """
        x = x.to(dtype=self.dtype)
        x = self.tokenConv(x)
        return x


class PatchEmbedding(nn.Module):
    """
    A module to embed time series data into patches.
    """

    def __init__(self, d_model, patch_len, stride, dropout, dtype=torch.float32):
        super(PatchEmbedding, self).__init__()
        # Patching parameters
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))  # Padding on the right

        # Token embedding layer
        self.value_embedding = TokenEmbedding(patch_len, d_model, dtype=dtype)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        self.dtype = dtype

    def forward(self, x):
        """
        Forward pass for PatchEmbedding.

        Args:
            x (Tensor): Shape [batch_size, n_vars, seq_len]

        Returns:
            Tensor: Shape [batch_size, L, d_model], where L = n_vars * num_patches
            n_vars (int): Number of variables (time series channels)
        """
        x = x.to(dtype=self.dtype)
        n_vars = x.shape[1]  # Number of variables

        # Padding
        x = self.padding_patch_layer(x)  # Shape: [batch_size, n_vars, seq_len + stride]

        # Unfold to create patches
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # Shape: [batch_size, n_vars, num_patches, patch_len]
        num_patches = x.shape[2]  # Number of patches

        # Reshape for convolution
        x = x.contiguous().view(-1, self.patch_len, num_patches)  # Shape: [batch_size * n_vars, patch_len, num_patches]

        # Token embedding
        x = self.value_embedding(x)  # Shape: [batch_size * n_vars, d_model, num_patches]

        # Permute to have num_patches in sequence dimension
        x = x.permute(0, 2, 1)  # Shape: [batch_size * n_vars, num_patches, d_model]

        # Reshape to combine batch_size and n_vars
        x = x.contiguous().view(-1, n_vars * num_patches, x.shape[-1])  # Shape: [batch_size, L, d_model], L = n_vars * num_patches

        return self.dropout(x), n_vars, num_patches  # Return number of patches for later use


class timeLLM(nn.Module):
    """
    A text-time series dual-modality model integrating a frozen LLM with time series data.

    Args:
        llm_name (str): Name of the pre-trained LLM to use.
        pred_len (int): Prediction length (number of future time steps to forecast).
        seq_len (int): Input sequence length of the time series.
        token_dim (int): Embedding dimension of the LLM.
        patch_len (int): Length of each time series patch.
        stride (int): Stride for patching the time series.
        d_model (int): Dimension of the model's internal representations.
        dropout (float): Dropout rate.
        n_time (int): Number of input features (time series variables).
        n_heads (int): Number of attention heads in the reprogramming layer.
        d_ff (int): Number of hidden neurons used in the last layer of LLM (defaults to using all).
        dtype (torch.dtype): Data type for computations.
    """

    def __init__(
        self, llm_name, pred_len, seq_len, n_time, token_dim, patch_len, stride, d_model, dropout=0, n_heads=8, d_ff=-1, dtype=torch.bfloat16
    ):
        super(timeLLM, self).__init__()
        self.pred_len = pred_len  # Prediction length
        self.seq_len = seq_len  # Input sequence length
        self.d_ff = d_ff
        self.token_dim = token_dim  # Embedding dimension of the LLM
        self.patch_len = patch_len  # Patch length
        self.stride = stride  # Stride for patching
        self.llm_name = llm_name  # Name of the LLM
        self.dtype = dtype

        # Load or download the pre-trained LLM
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models")
        os.makedirs(model_dir, exist_ok=True)
        try:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                f"{model_dir}/{llm_name}", trust_remote_code=True, local_files_only=True, torch_dtype=self.dtype
            )
        except EnvironmentError:
            print("Local model files not found. Attempting to download...")
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                f"{model_hf[llm_name]}", trust_remote_code=True, local_files_only=False, torch_dtype=self.dtype
            )

        # Load or download the tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                f"{model_dir}/{llm_name}", cache_dir=f"{model_dir}/{llm_name}", trust_remote_code=True, local_files_only=True
            )
        except EnvironmentError:
            print("Local tokenizer files not found. Attempting to download them...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                f"{model_hf[llm_name]}", cache_dir=f"{model_dir}/{llm_name}", trust_remote_code=True, local_files_only=False
            )

        # Handle padding token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = "[PAD]"
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
            self.tokenizer.pad_token = pad_token

        # Freeze LLM parameters
        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(d_model, self.patch_len, self.stride, dropout, dtype=self.dtype)

        # Extract embeddings from the LLM
        self.embeddings = self.llm_model.get_input_embeddings().weight  # Shape: [vocab_size, d_llm]
        self.embeddings = self.embeddings.to(dtype=self.dtype)
        self.vocab_size = self.embeddings.shape[0]
        self.num_prototype = 1000  # Number of text prototypes (V' << V)

        # Linear probing to obtain text prototypes E' from E
        self.prototype = nn.Linear(self.vocab_size, self.num_prototype, dtype=self.dtype)

        # Reprogramming layer
        self.reprogramming_layer = ReprogrammingLayer(d_model, n_heads, self.d_ff, self.token_dim, dtype=self.dtype)

        # Calculate number of patches
        self.patch_nums = ((seq_len + stride - patch_len) // stride) + 1  # Number of patches

        # Output projection
        self.head_nf = self.d_ff * self.patch_nums  # Flattened dimension
        self.output_projection = FlattenHead(n_time, self.head_nf, self.pred_len, head_dropout=dropout, dtype=self.dtype)

        self.active_branch = "model"
        self.to(dtype=self.dtype)

    def forward(self, x_enc, prompt):
        """
        Forward pass for the model.

        Args:
            x_enc (Tensor): Time series input, shape [batch_size, seq_len, n_vars]
            prompt (str): Textual prompt

        Returns:
            Tensor: Forecasted output, shape [batch_size, pred_len, n_vars]
        """
        x_enc = x_enc.to(dtype=self.dtype)
        dec_out = self.forecast(x_enc, prompt)
        return dec_out[:, -self.pred_len :, :]

    def forecast(self, x_enc, prefix_prompt, suffix_prompt):
        """
        Performs the forecasting by integrating time series data with the LLM.

        Args:
            x_enc (Tensor): Time series input, shape [batch_size, seq_len, n_vars]
            prompt (str): Textual prompt

        Returns:
            Tensor: Forecasted output, shape [batch_size, pred_len, n_vars]
        """
        # Tokenize the prompt
        prefix_prompt_tokens = self.tokenizer(
            prefix_prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.tokenizer.model_max_length
        ).input_ids
        suffix_prompt_tokens = self.tokenizer(
            suffix_prompt, return_tensors="pt", padding=True, truncation=True,
            max_length=self.tokenizer.model_max_length
        ).input_ids
        prompt_tokens = prompt_tokens.to(device=x_enc.device)  # Shape: [batch_size, prompt_len]

        # Get prompt embeddings
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_tokens).to(dtype=self.dtype)
        # Shape: [batch_size, prompt_len, d_llm]

        # Obtain text prototypes E' by linear probing E
        source_embeddings = self.prototype(self.embeddings.permute(1, 0)).permute(1, 0)

        # Prepare time series input
        x_enc = x_enc.permute(0, 2, 1).contiguous()  # Shape: [batch_size, n_vars, seq_len]
        enc_out, n_vars, num_patches = self.patch_embedding(x_enc)
        # enc_out: [batch_size, L, d_model], L = n_vars * num_patches

        # Reprogram time series patches using text prototypes
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        # enc_out: [batch_size, L, d_llm]

        # Concatenate prompt and reprogrammed embeddings
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        # Shape: [batch_size, prompt_len + L, d_llm]

        # Pass through the frozen LLM
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        # Shape: [batch_size, prompt_len + L, d_llm]

        # Discard prompt outputs
        prompt_len = prompt_embeddings.shape[1]
        dec_out = dec_out[:, prompt_len:, :]  # Shape: [batch_size, L, d_llm]

        # Optionally reduce dimensionality
        dec_out = dec_out[:, :, : self.d_ff]  # Shape: [batch_size, L, d_ff]

        # Reshape for output projection
        dec_out = dec_out.view(-1, n_vars, num_patches, self.d_ff)  # Shape: [batch_size, n_vars, num_patches, d_ff]
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()  # Shape: [batch_size, n_vars, d_ff, num_patches]

        # Apply output pr
