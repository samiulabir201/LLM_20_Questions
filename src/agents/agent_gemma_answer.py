import sys
import time
import gc
import numpy as np
import torch
from utils.util import get_last_question, get_keyword, is_valid_answer
from Tuned_models.model_gemma import model_gemma_load
from Tuned_models.model_all import delete_all_models


def format_question_for_gemma(obs):
    """
    Format the last question from the observation into a prompt for the Gemma model.
    """
    question_raw = get_last_question(obs)
    keyword = get_keyword(obs)
    question_formatted = (
        f'The keyword is "{keyword}". {question_raw} Answer to the question above with "yes" or "no".'
    )
    return question_formatted


def calculate_probabilities(model, tokenizer, query_prompt):
    """
    Calculate probabilities for "yes" and "no" responses using the Gemma model.
    """
    try:
        # Prepare inputs for the model
        inputs = tokenizer(query_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        # Extract logits and compute probabilities
        logits = output.logits[0, -1, :]
        probs = torch.softmax(logits, dim=0).cpu().detach().numpy()
        return probs
    except Exception as e:
        print(f"Error during probability calculation: {e}", file=sys.stderr)
        raise e


def get_response_from_probabilities(probs, tokenizer):
    """
    Compute the final response ("yes" or "no") based on token probabilities.
    """
    words = ["yes", "Yes", "no", "No"]
    prob_words = []

    try:
        for word in words:
            idx = tokenizer.convert_tokens_to_ids(word)
            prob = probs[idx]
            prob_words.append(prob)
            print(f"Probability of '{word}': {prob:.4f}", file=sys.stderr)

        prob_yes = prob_words[0] + prob_words[1]
        prob_no = prob_words[2] + prob_words[3]

        return "yes" if prob_yes > prob_no else "no"
    except Exception as e:
        print(f"Error in response computation: {e}", file=sys.stderr)
        return "no"


def get_answer_from_gemma(model, tokenizer, question_formatted):
    """
    Generate an answer from the Gemma model for the formatted question.
    """
    print(f"Formatted question: {question_formatted}", file=sys.stderr)

    try:
        probs = calculate_probabilities(model, tokenizer, question_formatted)
        print("Successfully calculated probabilities.", file=sys.stderr)
        return get_response_from_probabilities(probs, tokenizer)
    except Exception as e:
        print(f"Error in Gemma model processing: {e}", file=sys.stderr)
        return "no"


def agent_gemma_answer(obs, cfg):
    """
    Main function for the Gemma agent to generate answers for "answer" turns.
    """
    print("Starting Gemma agent...", file=sys.stderr)

    # Load the Gemma model
    delete_all_models("gemma")
    model, tokenizer = model_gemma_load()
    assert model is not None, "Failed to load Gemma model."
    assert tokenizer is not None, "Failed to load tokenizer."

    # Format the question and get the response
    question_formatted = format_question_for_gemma(obs)
    answer = get_answer_from_gemma(model, tokenizer, question_formatted)

    # Validate and normalize the response
    answer = answer.lower()
    if not is_valid_answer(answer):
        answer = "no"

    print(f"Gemma agent finished. Final answer: {answer}", file=sys.stderr)

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleared.", file=sys.stderr)

    return answer
