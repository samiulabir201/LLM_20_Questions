import os
import sys
import torch
import traceback

# Disable specific optimizations for compatibility
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# Set paths for Kaggle environment
KAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"
KAGGLE_SUBMISSION_LIB_PATH = "/kaggle/working/submission/lib"

def setup_paths():
    """Set up system paths for importing modules depending on the environment."""
    if os.path.exists(KAGGLE_AGENT_PATH):
        sys.path.insert(0, os.path.join(KAGGLE_AGENT_PATH, "lib"))
        sys.path.insert(0, os.path.join(KAGGLE_AGENT_PATH, "utils"))
        sys.path.insert(0, KAGGLE_AGENT_PATH)
    else:
        sys.path.insert(0, KAGGLE_SUBMISSION_LIB_PATH)

# Ensure proper paths are added
setup_paths()

# Import the main agent management function
try:
    from .agents.agent_manager import agent_manager
except ImportError:
    from .agents.agent_manager import agent_manager

def agent_fn(obs, cfg):
    """
    Main function hook for the agent.

    Args:
        obs (dict): Observation of the current state.
        cfg (dict): Configuration settings.

    Returns:
        str: The agent's response based on the observation.
    """
    print("#" * 120)
    print(f"turnType: {obs.get('turnType')}")
    
    try:
        # Delegate to the agent manager
        response = agent_manager(obs, cfg)
    except Exception as e:
        print(f"Error encountered: {e}")
        traceback.print_exc()
        response = "no"

    print(f"response: {response}")
    print("#" * 120)

    # Validate the response based on the turn type
    if obs.get("turnType") == "answer":
        if response not in ["yes", "no"]:
            response = "no"
    elif response is None:
        response = "Error in response."

    return response
