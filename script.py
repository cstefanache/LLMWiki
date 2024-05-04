import json
import os
import binascii
import re
import shutil
from llama_cpp import Llama
from huggingface_hub import hf_hub_download, scan_cache_dir

cache_dir = "./cached"
config = json.load(open("config.json"))

start_from = config["startFrom"]
max_items = config["maxItems"]
depth = config["depth"]
output_to = config["outputTo"]


def query_llm(
    llm,
    instruction,
    max_tokens=4096,
    top_k=40,
    seed=0,
    temperature=0.0,
    echo=True,
    skip_cache=False,
):
    cache_key = binascii.crc32(instruction.encode())
    cache_file = os.path.join(cache_dir, f"{cache_key}.txt")
    result = ""

    if os.path.exists(cache_file) and not skip_cache:
        with open(cache_file, "r") as f:
            result = f.read()
    else:
        llm_response = llm(
            instruction,
            max_tokens=max_tokens,
            top_k=top_k,
            seed=seed,
            temperature=temperature,
            repeat_penalty=1.3,
            stream=True,
        )

        for chunk in llm_response:
            val = chunk["choices"][0]["text"]
            result += val
            print(val, end="", flush=True)

        result = result.replace("▅", "").strip()
        with open(cache_file, "w") as f:
            f.write(result)

    if echo:
        print(instruction)
        print("###############")
        print(result)
    return result


def get_filename_from_complex_topic(filename: str) -> str:
    return (
        filename.replace(" ", "_")
        .replace(">", "_")
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("?", "_")
        .replace("*", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def go_into_list(llama, topic, depth, parent=None):
    print(f"Generating list for {topic}. Current depth: {depth}")
    # instruction = f"[INST] Should {topic} topic be broken into sub-sections? Start answer with yes or no. [/INST]"
    # response = query_llm(llama, instruction, max_tokens=32, echo=True)

    # check if "yes" is present in the first 10 characters in response
    if depth == 0:
        instruction = f"[INST] English text only. Generate a single page on the subject {topic} in markdown. [/INST]"
        response = query_llm(llama, instruction, echo=False)

        # Save the list to a file
        with open(f"{output_to}/{get_filename_from_complex_topic(topic)}.md", "w") as f:
            f.write(response)
        return topic, f"{output_to}/{get_filename_from_complex_topic(topic)}.md", "w"

    else:
        instruction = f"""
[INST]
<<SYS>>Your task is to generate a numbered list of topics on the provided subject, first level only, no sub-lists. The list should contain a maximum of {max_items} items. English only.<<SYS>>
Give me a numbered list of topics on the subject of "{topic}" that can be broken down into sub-sections.
[/INST]
"""
        response = query_llm(llama, instruction, echo=True)
        
        # # Extract the list
        list_pattern = re.compile(r"^\d+\.\s(.*)$", re.MULTILINE)
        list_matches = list_pattern.findall(response)
        links = []
        if depth > 0:
            for item in list_matches:
                links.append(
                    go_into_list(
                        llama,
                        item,
                        depth - 1,
                        parent=parent + " > " + topic if parent else topic,
                    )
                )

        # Save the list to a file
        with open(f"{output_to}/{get_filename_from_complex_topic(topic)}.md", "w") as f:
            f.write(f"# {topic}\n\n")
            for link in links:
                f.write(f"- [{link[0]}]({link[1]})\n")
        return topic, f"{output_to}/{get_filename_from_complex_topic(topic)}.md"


def main():
    if not os.path.exists("./cached"):
        os.makedirs("./cached")

    shutil.rmtree(f"./{output_to}", ignore_errors=True)
    os.makedirs(f"./{output_to}")

    # Download the model from the Hugging Face Hub
    model_path = hf_hub_download(
        repo_id=config["repository"], filename=config["model"], cache_dir="./models"
    )

    # Load the model
    llama = Llama(
        str(model_path),
        do_sample=True,
        num_return_sequences=1,
        n_ctx=4096,
        n_gpu_layers=64,
        n_threads=8,
        verbose=False,
    )

    go_into_list(llama, start_from, depth)


if __name__ == "__main__":
    main()
