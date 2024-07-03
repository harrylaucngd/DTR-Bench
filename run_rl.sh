export PYTHONPATH=$PYTHONPATH:/home/liuhx/DTR-Bench
python DTRBench/run_RL/online_discrete_search.py

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
                    self.llm_model = LlamaModel.from_pretrained(
                        f'{model_dir}/{configs.llm_model}',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llm_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = LlamaModel.from_pretrained(
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
                    self.llm_model = LlamaModel.from_pretrained(
                        f'{model_dir}/{configs.llm_model}',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llm_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = LlamaModel.from_pretrained(
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
            if configs.llm_model == 'llama-2-13b':
                self.llm_config = LlamaConfig.from_pretrained(f'{model_dir}/llama-2-13b')
                self.llm_config.num_hidden_layers = configs.llm_layers
                self.llm_config.output_attentions = True
                self.llm_config.output_hidden_states = True
                
                try:
                    self.llm_model = LlamaModel.from_pretrained(
                        f'{model_dir}/llama-2-13b',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llm_config,
                    )
                except EnvironmentError:
                    print("Local model files not found. Please ensure the model is correctly placed in ./model_hub")

                try:
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        f'{model_dir}/llama-2-13b',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:
                    print("Local tokenizer files not found. Please ensure the tokenizer is correctly placed in ./model_hub")
            
            elif configs.llm_model == 'llama-13b':
                
                self.llm_config = LlamaConfig.from_pretrained(f'{model_dir}/llama-13b')
                self.llm_config.num_hidden_layers = configs.llm_layers
                self.llm_config.output_attentions = True
                self.llm_config.output_hidden_states = True
                
                try:
                    self.llm_model = LlamaModel.from_pretrained(
                        f'{model_dir}/llama-13b',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llm_config,
                    )
                except EnvironmentError:
                    print("Local model files not found. Please ensure the model is correctly placed in ./model_hub")

                try:
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        f'{model_dir}/llama-13b',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:
                    print("Local tokenizer files not found. Please ensure the tokenizer is correctly placed in ./model_hub")
            
            elif configs.llm_model == 'llama-3-8b':

                
                self.llm_config = LlamaConfig.from_pretrained(f'{model_dir}/llama-3-8b')
                self.llm_config.num_hidden_layers = configs.llm_layers
                self.llm_config.output_attentions = True
                self.llm_config.output_hidden_states = True
                
                try:
                    self.llm_model = LlamaModel.from_pretrained(
                        f'{model_dir}/llama-3-8b',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llm_config,
                    )
                except EnvironmentError:
                    print("Local model files not found. Please ensure the model is correctly placed in ./model_hub")

                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        f'{model_dir}/llama-3-8b',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:
                    print("Local tokenizer files not found. Please ensure the tokenizer is correctly placed in ./model_hub")
            
            elif configs.llm_model == 'llama-2-7b':

                
                self.llm_config = LlamaConfig.from_pretrained(f'{model_dir}/llama-2-7b')
                self.llm_config.num_hidden_layers = configs.llm_layers
                self.llm_config.output_attentions = True
                self.llm_config.output_hidden_states = True
                
                try:
                    self.llm_model = LlamaModel.from_pretrained(
                        f'{model_dir}/llama-2-7b',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llm_config,
                    )
                except EnvironmentError:
                    print("Local model files not found. Please ensure the model is correctly placed in ./model_hub")

                try:
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        f'{model_dir}/llama-2-7b',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:
                    print("Local tokenizer files not found. Please ensure the tokenizer is correctly placed in ./model_hub")
            
            elif configs.llm_model == 'llama-7b':

                
                self.llm_config = LlamaConfig.from_pretrained(f'{model_dir}/llama-7b')
                self.llm_config.num_hidden_layers = configs.llm_layers
                self.llm_config.output_attentions = True
                self.llm_config.output_hidden_states = True
                
                try:
                    self.llm_model = LlamaModel.from_pretrained(
                        f'{model_dir}/llama-7b',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llm_config,
                    )
                except EnvironmentError:
                    print("Local model files not found. Please ensure the model is correctly placed in ./model_hub")

                try:
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        f'{model_dir}/llama-7b',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:
                    print("Local tokenizer files not found. Please ensure the tokenizer is correctly placed in ./model_hub")

            elif configs.llm_model == 'gpt2':
                self.llm_config = GPT2Config.from_pretrained(f'{model_dir}/gpt2')
                self.llm_config.num_hidden_layers = configs.llm_layers
                self.llm_config.output_attentions = True
                self.llm_config.output_hidden_states = True

                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        f'{model_dir}/gpt2',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llm_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.llm_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        f'{model_dir}/gpt2',
                        cache_dir=f'{model_dir}/gpt2',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2',
                        cache_dir=f'{model_dir}/gpt2',
                        trust_remote_code=True,
                        local_files_only=False
                    )