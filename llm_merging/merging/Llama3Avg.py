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
                "s50227harry/llama-3-8B-lora",
                None,
                #"abcdabcd987/gsm8k-llama2-7b-lora-16",
                #"636b5eb8da724edae406ba69ef90fd06478e6df7",
            ),
            (   "zjunlp/llama3-8b-iepile-lora",
                None,
                #"FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
                #"69f77190315afdb03a889d89bf2a0f932b311617",
            ),
        ]

        # Hyperparameters
        self.base_model_name = "meta-llama/Meta-Llama-3-8B"
        
        # We recommend specifying a revision id to ensure the model was not modified after May 31
        #self.base_model_revision_id = "01c7f73d771dfac7d292323805ebc428287df4f9"

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

    # Implement merge function
    def merge(
        self,
    ):

        """
        1) Load HuggingFace checkpoints and configs
        """
        super()._load_huggingface_models_and_configs()
        """
        2) Merge checkpoints  
        """
        parameter_lambdas = [0.8, 0.2]

        # Get individual models
        all_models = list(self.loaded_models.values())

        # Get all the parameters names (uses the first model and assume all the models have the same parameter)
        all_parameter_names = all_models[0].keys()
        torch.cuda.empty_cache()
        # Merge the models
        max_lora_rank = max(model['base_model.model.model.layers.0.mlp.down_proj.lora_A.weight'].shape[0] for model in all_models)

        for parameter_name in all_parameter_names:
            merged_parameter = None
            for parameter_lambda, model in zip(parameter_lambdas, all_models):
                parameter = model[parameter_name]
                
                # Handle LoRA rank mismatch
                if 'lora_A.weight' in parameter_name:
                    current_rank = parameter.shape[0]
                    if current_rank < max_lora_rank:
                        padding = torch.zeros(max_lora_rank - current_rank, parameter.shape[1], device=parameter.device)
                        parameter = torch.cat([parameter, padding], dim=0)
                elif 'lora_B.weight' in parameter_name:
                    current_rank = parameter.shape[1]
                    if current_rank < max_lora_rank:
                        padding = torch.zeros(parameter.shape[0], max_lora_rank - current_rank, device=parameter.device)
                        parameter = torch.cat([parameter, padding], dim=1)

                if merged_parameter is None:
                    merged_parameter = parameter * parameter_lambda
                else:
                    merged_parameter += parameter * parameter_lambda

            self.merged_model[parameter_name] = merged_parameter
        torch.cuda.empty_cache()
        """
        3) Load base model and tokenizer
        """
        self._load_base_model()
        self.base_model.gradient_checkpointing_enable()
        self._load_tokenizer()

        """
        4) Load merged model into base model 
        """
        # Modify the base model. This is needed for Peft, which wraps the base_model in a Peft wrapper.
        huggingface_config = list(self.loaded_configs.values())[0]
        if huggingface_config is not None:
            self.base_model = get_peft_model(self.base_model, huggingface_config)
            set_peft_model_state_dict(self.base_model, self.merged_model)

        else:
            self.base_model.load(self.merged_model)

        # Requires to make results deterministic. If not set, we will just run once and use the results from the first pass.
        self.base_model.eval()

        return self.base_model
