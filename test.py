from transformers import LlamaConfig, LlamaModel, LlamaTokenizer

model_dir = './model_hub'
            
llama_config = LlamaConfig.from_pretrained(f'{model_dir}/llama-2-13b')

llm_model = LlamaModel.from_pretrained(
    f'{model_dir}/llama-2-13b',
    trust_remote_code=True,
    local_files_only=True,
    config=llama_config,
)