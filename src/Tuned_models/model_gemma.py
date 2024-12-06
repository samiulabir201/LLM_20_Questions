import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
    set_seed,
)
from utils.util import (
    is_local,
    is_kaggle_notebook,
    is_kaggle_agent,
    KAGGLE_AGENT_PATH,
    KAGGLE_NOTEBOOK_PATH,
    clear_cache,
)

# Set seed for reproducibility
set_seed(42)

# Define model path based on environment
MODEL_PATH = (
    "meta-llama/Meta-Llama-3-8B-Instruct"
    if is_local()
    else (
        KAGGLE_NOTEBOOK_PATH / "meta-llama-3-8b-instruct-quant-8bit"
        if is_kaggle_notebook()
        else KAGGLE_AGENT_PATH / "meta-llama-3-8b-instruct-quant-8bit"
    )
)

# Quantization level
QUANT = 8  # Supported levels: 4, 8
model = None
tokenizer = None


def get_device():
    """
    Get the current device of the model.
    """
    return "cuda" if "cuda" in str(model.device) else "cpu"


def move_model_to_device(target_device):
    """
    Move the model to the specified target device.
    """
    global model
    if model is None or QUANT is not None or get_device() == target_device:
        return model

    model.to(target_device)
    return model


def delete_model():
    """
    Clear the loaded model and tokenizer to free memory.
    """
    global model, tokenizer
    model = None
    tokenizer = None
    clear_cache()


def model_to_gpu():
    """
    Move the model to GPU.
    """
    return move_model_to_device("cuda")


def model_to_cpu():
    """
    Move the model to CPU.
    """
    return move_model_to_device("cpu")


def load_quantization_config():
    """
    Generate the appropriate quantization configuration based on the specified level.
    """
    if QUANT == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif QUANT == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_has_fp16_weight=False,
        )
    else:
        raise ValueError("Unsupported quantization level. Only 4-bit and 8-bit are allowed.")


def load_model_and_tokenizer():
    """
    Load and configure the model and tokenizer.
    """
    print("Loading model configuration...")
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.gradient_checkpointing = True

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    quantization_config = load_quantization_config()

    print(f"Loading model with {QUANT}-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        load_in_8bit=(QUANT == 8),
        config=config,
        device_map="cuda:0" if QUANT == 8 else "auto",
        torch_dtype="auto",
    )

    print("Model and tokenizer successfully loaded.")
    return model, tokenizer


def model_gemma_load():
    """
    Load the Gemma model and tokenizer if not already loaded.
    """
    global model, tokenizer

    if model is not None:
        model_to_gpu()
        return model, tokenizer

    print("Initializing Gemma model loading...")
    print(f"Is local: {is_local()}")
    print(f"Is Kaggle Notebook: {is_kaggle_notebook()}")
    print(f"Is Kaggle Agent: {is_kaggle_agent()}")

    print(f"Model path: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")

    # Disable incompatible optimizations
    print("Disabling memory-efficient SDP and flash SDP...")
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    return model, tokenizer
