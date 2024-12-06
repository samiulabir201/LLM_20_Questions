import os
import re
import sys
import subprocess
import gc
import transformers
import torch

from utils.util import get_last_question, get_keyword, is_valid_answer, is_local
from Tuned_models.model_deepmath import model_deepmath_load
from Tuned_models.model_all import delete_all_models


def post_process(output):
    """
    Post-process the model's raw output into a 'yes' or 'no' response.
    """
    if output is None:
        return None
    try:
        output = re.sub(r"[^a-zA-Z]", "", output).lower()
        if output in ["yes", "true", "1"]:
            return "yes"
        elif output in ["no", "false", "0"]:
            return "no"
        return None
    except Exception as e:
        raise ValueError(f"Unexpected output: {output}") from e


def process_output(output, obs=None, n_try=None):
    """
    Process raw model output into final results and code execution results.
    """
    if not is_local():
        print("Raw output:", output)

    code_file = "code.py"
    try:
        if obs:
            n_turn = len(obs["questions"])
            os.makedirs("code", exist_ok=True)
            code_file = f"code/{n_turn:02}_{n_try}.py"
            os.makedirs("raw_output", exist_ok=True)
            with open(f"raw_output/{n_turn:02}_{n_try}.txt", "w") as fout:
                fout.write(output)

        # Extract and save Python code
        code = output.split("```")[1][7:]
        with open(code_file, "w") as fout:
            fout.write(code)

        # Run the code and capture output
        batcmd = f"timeout 2 {sys.executable} {code_file}"
        shell_output = subprocess.check_output(batcmd, shell=True).decode("utf8")
        code_output = post_process(shell_output)
    except Exception as e:
        print(f"Error processing code: {e}")
        code_output = None

    try:
        # Extract boxed result from raw output
        result_output = re.findall(r"\\boxed\{(.*)\}", output)
        result_output = result_output[-1] if result_output else None
    except Exception as e:
        print(f"Error extracting result: {e}")
        result_output = None

    code_output = post_process(code_output)
    result_output = post_process(result_output)

    print(f"Final code output: {code_output}")
    print(f"Final result output: {result_output}")

    return result_output, code_output


def format_question_from_deepmath(obs):
    """
    Format the last question from the observation into a prompt for DeepMath.
    """
    question_raw = get_last_question(obs)
    keyword = get_keyword(obs).lower()

    # Normalize letter cases in the question
    for i in range(26):
        c_lower, c_upper = chr(ord("a") + i), chr(ord("A") + i)
        question_raw = question_raw.replace(f'"{c_upper}"', f'"{c_lower}"')
        question_raw = question_raw.replace(f"'{c_upper}'", f"'{c_lower}'")

    # Build formatted question
    question_formatted = (
        f'The keyword is "{keyword}" (as a whole, so do not sort or split it). '
        + question_raw.replace(" sorting ", " lexicographical ")
    )
    return question_formatted


def get_answer_from_deepmath(model, tokenizer, obs, cfg, question_formatted, n_try):
    """
    Query DeepMath with a formatted question and process the response.
    """
    keyword = get_keyword(obs).lower()

    # Build the input prompt
    tool_instruction = (
        "\nPlease integrate step-by-step natural language reasoning with programs "
        "to solve the problem above, and put your final answer within \\boxed{}."
    )
    messages = [{"role": "user", "content": question_formatted + tool_instruction}]
    query_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    # Add a Python code header for programmatic reasoning
    output_head = (
        "```python\n"
        "def solve():\n"
        f'    """{question_formatted}"""\n'
        f'    keyword = "{keyword}"\n'
        "    "
    )
    query_prompt += output_head

    # Generate response
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype="auto",
        device_map="auto",
    )
    raw_output = pipeline(
        query_prompt,
        max_new_tokens=1536,
        do_sample=True,
        temperature=0.7,
        return_full_text=False,
        stop_strings=["```output"],
    )[0]["generated_text"]
    raw_output = output_head + raw_output

    # Process the raw output
    return process_output(raw_output, obs, n_try)[1]  # Return code output


def agent_deepmath_answer(obs, cfg):
    """
    Main function for DeepMath agent to handle 'answer' turns.
    """
    print("Starting DeepMath agent...")

    # Load the DeepMath model
    delete_all_models("deepmath")
    model, tokenizer = model_deepmath_load()
    assert model is not None and tokenizer is not None

    question_formatted = format_question_from_deepmath(obs)
    print(f"Formatted question: {question_formatted}")

    N_max = 5
    ans_dict = {}

    # Query the model up to N_max times to ensure reliability
    for n in range(N_max):
        try:
            ans = get_answer_from_deepmath(model, tokenizer, obs, cfg, question_formatted, n)
            if ans:
                ans_dict[ans] = ans_dict.get(ans, 0) + 1
                if ans_dict[ans] > N_max // 2:
                    break
        except Exception as e:
            print(f"Error in DeepMath query: {e}")

    print(f"Answer dictionary: {ans_dict}")

    # Determine the final answer
    ans = max(ans_dict, key=ans_dict.get) if ans_dict else "no"

    print(f"DeepMath agent finished. Final answer: {ans}")

    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    return ans
