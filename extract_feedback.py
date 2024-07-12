import datasets
from torch import cuda
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
import argparse
import os
import uuid
import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from spacy_langdetect import LanguageDetector
from spacy.language import Language
import spacy
import re
import json
from collections import Counter
import seaborn as sns
import numpy as np

nlp = spacy.load("en_core_web_sm")  # 1


def get_lang_detector(nlp, name):
    return LanguageDetector()


Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

CACHE_DIR = "Insert Your Cache Directory Here"
TOKEN = "Insert Your Token Here"

model_format_dict = {"mistralai/Mixtral-8x7B-Instruct-v0.1": {"prefix": "<s>[INST]", "suffix": "[/INST]"},
                     "01-ai/Yi-34B-Chat": {"prefix": "<s>[INST] ", "suffix": " [/INST]"}}

feedback_categories_to_abbr = {"Repeat or Rephrase": "UR1",
                               "Make Aware with Correction": "UR2",
                               "Make Aware without Correction": "UR3",
                               "Ask for Clarification": "UR4",
                               "Positive Feedback": "UR5",
                               "Repeat or Rephrase (UR1)": "UR1",
                               "Make Aware with Correction (UR2)": "UR2",
                               "Make Aware without Correction (UR3)": "UR3",
                               "Ask for Clarification (UR4)": "UR4",
                               "Positive Feedback (UR5)": "UR5",
                               "UR2": "UR2",
                               "UR3": "UR3",
                               "UR4": "UR4",
                               "UR5": "UR5",
                               "UR6": "UR6"}


def filter_non_eng_responses(indices, examples_with_no_prefix=None):
    if examples_with_no_prefix is None:
        examples_with_no_prefix = get_user_response_from_lmsys(indices, max_examples=max(indices) + 1)
    examples_to_keep = []
    indices_to_keep = []
    indices_to_remove = []
    for idx, example in zip(indices, examples_with_no_prefix):
        if idx in [127]:
            continue
        doc = nlp(example)
        if doc._.language["language"] == "en":
            indices_to_keep.append(idx)
            examples_to_keep.append(example)
        else:
            indices_to_remove.append(idx)
    print(f"non-english indices_to_remove: {indices_to_remove}")
    return indices_to_keep, examples_to_keep


def run_prompt(prompt, model, tokenizer, max_new_tokens=256):
    temperature = 0.2
    top_p = 0.95
    repetition_penalty = 1.0

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        # seed=42,
    )
    input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
    output = model.generate(input_ids, **generate_kwargs)
    model_response = tokenizer.decode(output[0], skip_special_tokens=True)
    prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    commonprefix = os.path.commonprefix([prompt, model_response])
    model_response = model_response[len(commonprefix):]
    return model_response


def parse_category(category):
    for key in feedback_categories_to_abbr:
        if key in category:
            return feedback_categories_to_abbr[key]
    return None


def parse_formatted_answer(answer, prompt):
    answer = answer.strip()
    lines = answer.split("\n")
    feedbacks = []
    for line in lines:
        if "-" in line:
            category = parse_category(line.strip())
            if category:
                match = SequenceMatcher(None, prompt.lower(), line.lower()).find_longest_match(0, len(prompt), 0, len(line))  # find the longest common substring
                if match.size < 5:
                    print(match)
                feedbacks.append((category, line[match.b:match.b + match.size].strip()))
    return feedbacks


def parse_formatted_answer_json(answer, prompt):
    jsons_found = re.findall(r'\{[^{}]*\}', answer)
    feedbacks = []
    confidences = []
    for json_answer in jsons_found:
        if "User Response Pattern" in json_answer and "User Response Text" in json_answer:
            try:
                json_answer = json_answer.replace("\\","")
                json_answer = json.loads(json_answer)
                category = parse_category(json_answer["User Response Pattern"])
                text = json_answer["User Response Text"]
                keep = True
                if "Confidence Level (1-5)" in json_answer:
                    if float(json_answer["Confidence Level (1-5)"]) < 5:
                        print(json_answer)
                        print(prompt)
                        # keep = False
                    confidences.append(json_answer["Confidence Level (1-5)"])
                match = SequenceMatcher(None, prompt.lower(), text.lower()).find_longest_match(0, len(prompt), 0, len(text))
                # if match.size < 5:
                #     print(match)
                if category and text in prompt and keep:
                    feedbacks.append((category, text[match.b:match.b + match.size].strip()))
            except Exception as e:
                print(e)
                print(json_answer)
    return feedbacks, confidences


def parse_formatted_answer_judge(answer, prompt):
    jsons_found = re.findall(r'\{[^{}]*\}', answer)
    feedbacks = []
    confidences = []
    for json_answer in jsons_found:
        if "User Response Satisfaction (1-5)" in json_answer and "User Response Text" in json_answer:
            try:
                json_answer = json_answer.replace("\\","")
                json_answer = json.loads(json_answer)
                user_statisfaction = float(json_answer["User Response Satisfaction (1-5)"])
                text = json_answer["User Response Text"]
                keep = True
                match = SequenceMatcher(None, prompt.lower(), text.lower()).find_longest_match(0, len(prompt), 0, len(text))
                # if match.size < 5:
                #     print(match)
                if user_statisfaction and text in prompt and keep:
                    feedbacks.append((user_statisfaction, text[match.b:match.b + match.size].strip()))
            except Exception as e:
                print(e)
                print(json_answer)

    return feedbacks


def load_lmsys(max_examples=100000):
    # load the dataset and prepare fields
    lmsys_dataset = datasets.load_dataset("lmsys/lmsys-chat-1m", cache_dir=CACHE_DIR,
                                          token=TOKEN)
    lmsys_dataset = lmsys_dataset['train'].select(range(min(max_examples * 10, 1000000)))  # to make things faster
    pd_dataset = lmsys_dataset.to_pandas()
    pd_dataset["iterations"] = pd_dataset["conversation"].apply(lambda x: len(x))
    pd_dataset = pd_dataset[pd_dataset["iterations"] > 3]
    return pd_dataset


def prepare_examples_from_lmsys(indices, prompt_prefix="", max_examples=100000, model_id=None):
    pd_dataset = load_lmsys(max_examples)

    if pd_dataset.shape[0] < max(indices):
        print("Warning: the dataset is too small for the requested indices")
        new_indices = [i for i in indices if i < pd_dataset.shape[0]]
        removed_indices = [i for i in indices if i >= pd_dataset.shape[0]]
        print("Removing the following indices:", removed_indices)
        indices = new_indices

    # collect the examples
    examples = []
    for i in indices:
        example = prompt_prefix
        conv = pd_dataset.iloc[i]["conversation"]
        for reply in conv:
            if reply['role'] == 'user':
                example += f"# user: {reply['content']}\n"
            if reply['role'] == 'assistant':
                example += f"# assistant: {reply['content']}\n"
        if model_id in model_format_dict:
            example = model_format_dict[model_id]["prefix"] + example + model_format_dict[model_id]["suffix"]
        examples.append(example)
    return examples, indices


def get_user_response_from_lmsys(indices, max_examples=100000):
    pd_dataset = load_lmsys(max_examples)
    responses = []

    if pd_dataset.shape[0] < max(indices):
        print("Warning: the dataset is too small for the requested indices")
        new_indices = [i for i in indices if i < pd_dataset.shape[0]]
        removed_indices = [i for i in indices if i >= pd_dataset.shape[0]]
        print("Ignoring the following indices:", removed_indices)
        indices = new_indices

    for i in indices:
        example = ""
        conv = pd_dataset.iloc[i]["conversation"]
        for reply in conv:
            if reply['role'] == 'user':
                example += f'{reply["content"]}\n'
        responses.append(example)
    return responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--indices_file", type=str, default=None)
    parser.add_argument("--prompt_prefix_file", type=str, default=None)
    parser.add_argument("--existing_exp_dir", type=str, default=None)
    parser.add_argument("--reparse_answers", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--data_size", type=int, default=-1)
    parser.add_argument("--continue_experiment", action="store_true")
    args = parser.parse_args()

    # read the prompt prefix file
    if args.prompt_prefix_file is not None and os.path.exists(args.prompt_prefix_file):
        with open(args.prompt_prefix_file, 'r') as f:
            guidelines_prompt = f.read()
    else:
        guidelines_prompt = "There are five different patterns in user responses subsequent to errors in system utterances:\n" \
                            "Ignore and Continue (UR1) - The user ignores the error and continues the conversation, e.g., Okay. Let’s leave it like that. \n" \
                            "Repeat or Rephrase (UR2) - The user repeats or rephrases their concern, e.g., Actually, I wanted ... \n" \
                            "Make Aware with Correction (UR3) - The user makes the system aware of the error and provides information to address what is missing or wrong in its utterance, e.g., No. I wanted you to ... \n" \
                            "Make Aware without Correction (UR4) - The user makes the system aware of the error without providing additional information, e.g., You’re wrong. \n" \
                            "Ask for Clarification (UR5) - The user asks for clarification, e.g., Are you sure? Is it really that ...\n" \
                            "\n" \
                            "Given these guidelines, please recognize such user responses in the following dialogue. If thet are not such, please say so:\n"

    if args.existing_exp_dir is not None and not args.continue_experiment:  # we already have the responses
        all_errors = []
        directory = os.fsencode(args.existing_exp_dir)
        filenames = [os.fsdecode(file) for file in os.listdir(directory)]

        if args.reparse_answers:  # reparse the answers
            filenames = [filename for filename in filenames if "model_response" in filename]
            indices = [int(filename[(len("model_response_")):-(len(".txt"))]) for filename in filenames]
            indices.sort()
            examples_with_no_prefix = get_user_response_from_lmsys(indices, max_examples=max(indices) + 1)
            indices, examples_with_no_prefix = filter_non_eng_responses(indices, examples_with_no_prefix)
            confidence_scores = []
            for i, idx in enumerate(indices):
                print(f"parsing {idx}")
                with open(os.path.join(args.existing_exp_dir, f"model_response_{idx}.txt"), 'r') as f:
                    model_response = f.read()
                    if "json" in args.existing_exp_dir or "json" in args.prompt_prefix_file:
                        feedbacks, confi = parse_formatted_answer_json(model_response, examples_with_no_prefix[i])
                        confidence_scores += confi
                    else:
                        feedbacks = parse_formatted_answer(model_response, examples_with_no_prefix[i])
                    with open(os.path.join(args.existing_exp_dir, f"parsed_errors_{idx}.txt"), 'w+') as f:
                        for feedback in feedbacks:
                            f.write(f"{feedback[0]}-{feedback[1]}\n")
                    all_errors.append(feedbacks)
            confidence_scores = [float(score) for score in confidence_scores]
            print(Counter(confidence_scores))
            plt.hist(confidence_scores)
            plt.show()
            sns.histplot(sorted(confidence_scores), bins=5, alpha=0.5, label='Confidence Level', color="lightgreen")
            sns.despine()
        else:  # use the already parsed answers
            filenames = [filename for filename in filenames if "parsed_errors" in filename]
            indices = [int(filename[(len("parsed_errors_")):-(len(".txt"))]) for filename in filenames]
            indices.sort()
            if args.data_size > 0:
                indices = [idx for idx in indices if idx < args.data_size]
            # indices, _ = filter_non_eng_responses(indices)
            for idx in indices:
                with open(os.path.join(args.existing_exp_dir, f"parsed_errors_{idx}.txt"), 'r') as f:
                    feedbacks = []
                    for line in f:
                        line = line.strip()
                        where = line.find("-")
                        feedbacks.append((line[:where], line[where + 1:]) if where != -1 else (line, ""))
                    all_errors.append(feedbacks)

    else:  # we need to run the model to generate the responses
        indices = []  # read the indices file
        if args.indices_file is not None:
            with open(args.indices_file, 'r') as f:
                for line in f:
                    indices.append(int(line.strip()))
        else:
            indices = [i for i in range(args.data_size)]  # more!

        # remove the indices that already have responses
        if args.existing_exp_dir is not None and args.continue_experiment:
            directory = os.fsencode(args.existing_exp_dir)
            filenames = [os.fsdecode(file) for file in os.listdir(directory)]
            existing_indices = [int(filename[(len("model_response_")):-(len(".txt"))]) for filename in filenames if filename[(len("model_response_")):-(len(".txt"))].isdigit()]
            indices = [idx for idx in indices if idx not in existing_indices]

        indices.sort()
        # indices, _ = filter_non_eng_responses(indices)

        examples, indices = prepare_examples_from_lmsys(indices, prompt_prefix=guidelines_prompt, model_id=args.model_id)

        # load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=CACHE_DIR)

        if args.quantize:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=quantization_config,
                                                         device_map='auto',
                                                         cache_dir=CACHE_DIR,
                                                         token=TOKEN)
        else:
            # model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16,
            #                                              cache_dir=CACHE_DIR).to(device)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id,
                                                          cache_dir=CACHE_DIR,
                                                          token=TOKEN).to(device)

        # model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=quantization_config,
        #                                              device_map='auto',
        #                                              cache_dir=CACHE_DIR)

        # run the examples, save the responses and parse them
        model_responses = []
        all_errors = []
        timestamp_str = str(datetime.datetime.now()).replace(" ", "_")  # unique identifier for this run
        path = f"{args.model_id}_{args.prompt_prefix_file}"

        if args.existing_exp_dir is not None and args.continue_experiment:
            full_path = args.existing_exp_dir
        else:
            if args.quantize:
                path += "_quantized"
            dir_path_no_slash = os.path.join("model_responses", path.replace("/", "_"))
            if not os.path.exists(dir_path_no_slash):
                os.mkdir(dir_path_no_slash)
            full_path = os.path.join(dir_path_no_slash, timestamp_str)
            os.mkdir(full_path)

        examples_with_no_prefix = get_user_response_from_lmsys(indices, max_examples=max(indices) + 1)
        i = 0
        for idx, example in zip(indices, examples):
            print("=====================================================")
            try:
                model_response = run_prompt(example, model, tokenizer, args.max_new_tokens)
            except Exception as e:
                print(f"error in example {idx}: {e}")
                continue
            with open(os.path.join(full_path, f"model_response_{idx}.txt"), 'w+') as f:
                f.write(model_response)
            model_responses.append(model_response)

            if "json" in full_path or "json" in args.prompt_prefix_file:
                feedbacks, confi = parse_formatted_answer_json(model_response, examples_with_no_prefix[i])
                # confidence_scores += confi
            else:
                feedbacks = parse_formatted_answer(model_response, examples_with_no_prefix[i])
            i += 1
            with open(os.path.join(full_path, f"parsed_errors_{idx}.txt"), 'w+') as f:
                for feedback in feedbacks:
                    f.write(f"{feedback[0]}-{feedback[1]}\n")
            all_errors.append(feedbacks)

    print("mean num errors per conversation:", sum([len(errors) for errors in all_errors]) / len(all_errors) if (len(all_errors) > 0) else 0)
