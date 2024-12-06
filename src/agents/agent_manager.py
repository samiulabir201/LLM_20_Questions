from .alpha import agent_alpha
from .beta import agent_beta
from utils.util import QUESTION_IS_ALPHA

FLAG_USE_ALPHA = True

def agent_manager(obs, cfg):
    """Decides which sub-agent to call based on the game state."""

    if obs.turnType == "ask":
        response = None
        if FLAG_USE_ALPHA:
            if len(obs.questions) == 0:
                response = QUESTION_IS_ALPHA
            else:
                if obs.answers[0] == "yes":
                    response = agent_alpha(obs, cfg)
                else:
                    response = agent_beta(obs, cfg)
        if response is None:
            response = agent_beta(obs, cfg)

    elif obs.turnType == "guess":
        response = None
        if FLAG_USE_ALPHA and obs.answers[0] == "yes":
            response = agent_alpha(obs, cfg)
        if response is None:
            response = agent_beta(obs, cfg)

    elif obs.turnType == "answer":
        print(f"last question: {obs.questions[-1]}")
        assert len(obs.questions) > 0

        # Determine if the questioner is using Agent Alpha strategy
        questioner_is_alpha = False  # Adjust if needed

        if questioner_is_alpha:
            response = agent_alpha(obs, cfg)
        else:
            # Use LLM for answering
            response = agent_beta(obs, cfg)

        if response not in ["yes", "no"]:
            print(f"ERROR in response: {response}")
            response = "no"

    else:
        # Unexpected turnType
        assert False, f"Unexpected turnType: {obs.turnType}"

    return response
