import torch

from llm_merging.merging.Merges import Merges
from peft import get_peft_model, set_peft_model_state_dict

torch.cuda.empty_cache()


class Llama3Avg(Merges):
    def __init__(self, name):
        super().__init__(name)

        """
        These values are meant to be modified by the user.
        """
        
        # Give a list of models to load for the merge. Each element is the list a is a tuple of (model, revision_id). We recommend specifying a revision id to ensure the model was not modified after May 31
        self.list_models = [
            (
                #"s50227harry/llama-3-8B-lora",
                "meta-llama/Meta-Llama-3-8B-Instruct",
                None,
                # "abcdabcd987/gsm8k-llama2-7b-lora-16",
                # "636b5eb8da724edae406ba69ef90fd06478e6df7",
            ),
            (
                "zjunlp/llama3-8b-iepile-lora",
                None,
                # "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
                # "69f77190315afdb03a889d89bf2a0f932b311617",
            ),
        ]

        # Hyperparameters
        self.base_model_name = "meta-llama/Meta-Llama-3-8B"

        # We recommend specifying a revision id to ensure the model was not modified after May 31
        # self.base_model_revision_id = "01c7f73d771dfac7d292323805ebc428287df4f9"

        self.max_seq_len = None
        self.max_gen_len = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Architecture must match base model.
        self.architecture = "decoder"
        """
        These are variables used later in the code and not intended to be set, but feel free to adapt to your use case.  
        """
        # Loaded models and configs
        self.loaded_models = {}
        self.loaded_configs = {}

        # Merged model parameters
        self.merged_model = {}



    def merge(self):
        super()._load_huggingface_models_and_configs()
        
        all_models = list(self.loaded_models.values())
        all_parameter_names = all_models[0].keys()
        parameter_lambdas = [0.2, 0.8]
        
        # Determine target LoRA rank
        lora_a_keys = [key for key in all_parameter_names if 'lora_A.weight' in key]
        if lora_a_keys:
            target_lora_rank = min(model[lora_a_keys[0]].shape[0] for model in all_models)
        else:
            raise ValueError("No LoRA A weights found in the model")
        
        for parameter_name in all_parameter_names:
            merged_parameter = None
            for parameter_lambda, model in zip(parameter_lambdas, all_models):
                parameter = model[parameter_name]
                
                # Handle LoRA rank mismatch
                if 'lora_A.weight' in parameter_name and parameter.shape[0] > target_lora_rank:
                    parameter = parameter[:target_lora_rank, :]
                elif 'lora_B.weight' in parameter_name and parameter.shape[1] > target_lora_rank:
                    parameter = parameter[:, :target_lora_rank]
                
                if merged_parameter is None:
                    merged_parameter = parameter * parameter_lambda
                else:
                    merged_parameter += parameter * parameter_lambda
            
            self.merged_model[parameter_name] = merged_parameter
    
        # Load base model and apply merged LoRA weights
        self._load_base_model()
        self._load_tokenizer()
        
        # Modify PEFT config to match target LoRA rank
        peft_config = list(self.loaded_configs.values())[0]
        peft_config.r = target_lora_rank
        
        # Apply merged LoRA weights
        self.base_model = get_peft_model(self.base_model, peft_config)
        set_peft_model_state_dict(self.base_model, self.merged_model)
        
        return self.base_model