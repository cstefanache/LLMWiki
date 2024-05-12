import json
import os
import re
import binascii
import networkx as nx
import matplotlib.pyplot as plt
from llama_cpp import Llama
from huggingface_hub import hf_hub_download, scan_cache_dir

cache_dir = "./cached"
output_dir = "./output"
config = json.load(open("config.json"))

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

terms = config["terms"]


def query_llm(
    llm,
    instruction,
    max_tokens=4000,
    top_k=10,
    top_p=0.2,
    seed=0,
    temperature=0,
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
            top_p=top_p,
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


def main():
    if not os.path.exists("./cached"):
        os.makedirs("./cached")

    # Download the model from the Hugging Face Hub
    model_path = hf_hub_download(
        repo_id=config["repository"], filename=config["model"], cache_dir="../models"
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

    for index1, term1 in enumerate(terms):
        for index2, term2 in enumerate(terms):
            if term1 != term2 and index1 < index2:
                reponses = []

                print("#### Reasoning")
                instruction = f"[INST] Infer inspirations, relationships and commonalities between {term1} and {term2}. [/INST]"
                response = query_llm(llama, instruction, echo=False)
                reponses.append(response)

                print("#### Extract Relationships")
                instruction = f"<s>{response} [INST] Extract the  most relevant list of terms or concepts used in the provided content. [/INST]"
                response = query_llm(llama, instruction, echo=False)
                reponses.append(response)

                print("#### Extract Relationships")
                instruction = f"<s>{response} [INST] Find single word relationship type between all terms from the provided content and return a numbered list of pairs and their relationship. Try to include a term in multiple pairs and relationships. [/INST]"
                response = query_llm(llama, instruction, echo=False)
                reponses.append(response)


                # print("#### List")
                # instruction = f"{response}\m[INST] Extract all the terms and concepts listed in the provided text related to {term1} and {term2} or between them. Try to extract as many possible concepts and relationships as possible [/INST]"
                # response = query_llm(llama, instruction, echo=False)
                # reponses.append(response)

                # print("#### Rels")
                # instruction = f"{response}\m[INST] Based on the provided content generate a list of pairs and relationship types. [/INST]"
                # response = query_llm(llama, instruction, echo=False)
                # reponses.append(response)

                print("#### Graph")
                instruction = f"""
{response}
[INST]
Generate a dependency graph of terms and concepts based on the provided text.
IDs starts from 0. The first two nodes are {term1} and {term2}.
Output a single valid json, without comments, in the following strict format: 
```json
{"{"}
    "nodes": [
        {"{"}
            "id": "number",
            "label":"string"
        {"}"}
    ],
    "links": [
        {"{"}
            "source": "number",
            "target": "number",
            "relationship": "string"
        {"}"}
    ]
{"}"}
```

Use all the extracted terms and concepts and relationships to generate the graph.
[/INST]
"""
                response = query_llm(llama, instruction, echo=False)
                reponses.append("```\n" + instruction + "\n```")
                reponses.append(response)

                # extract all between ``` and ```
                response = response[
                    response.index("```json") + 7 : response.rindex("```")
                ]

                # remove everything between ``` ```
                response = re.sub(r"```.*?```", "", response, flags=re.DOTALL)

                # remove all comments from response
                response = re.sub(r"//.*", "", response)

                try:
                    data = json.loads(response)

                    # get all the keys that starts with nodes_ and merge them with nodes
                    nodes = data["nodes"]
                    for key in data.keys():
                        if key.startswith("nodes_"):
                            # nodes.update(data[key])
                            for node in data[key]:
                                if node not in nodes:
                                    nodes.append(node)

                    links = data["links"]
                    for key in data.keys():
                        if key.startswith("links_"):
                            for link in data[key]:
                                if link not in links:
                                    links.append(link)

                    data["nodes"] = nodes
                    data["links"] = links

                    plt.figure(figsize=(10, 10))

                    G = nx.json_graph.node_link_graph(data, directed=True)

                    # Draw the graph
                    pos = nx.shell_layout(G)  # positions for all nodes

                    # nodes
                    nx.draw_networkx_nodes(G, pos, node_size=300)

                    # edges
                    nx.draw_networkx_edges(
                        G, pos, arrowstyle="->", arrowsize=10, width=2
                    )

                    # labels
                    node_labels = nx.get_node_attributes(G, "label")
                    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7)

                    # edge labels
                    edge_labels = nx.get_edge_attributes(G, "relationship")
                    nx.draw_networkx_edge_labels(
                        G, pos, edge_labels=edge_labels, font_size=7
                    )

                    # save the graph
                    plt.tight_layout(pad=0)
                    plt.savefig(f"output/{term1}_{term2}.png")
                    # plt.show()

                    # generate a mermaid graph from data json
                    mermaid = "graph TD\n"
                    for node in data["nodes"]:
                        label = node["label"]
                        # strip inner () from label
                        if "(" in label:
                            label = label[: label.index("(")]
                        mermaid += f"{node['id']}({label})\n"

                    def make_relationship(source, target, relationship):
                        if relationship is None:
                            return f"{source} -->{target}\n"
                        else:
                            return f"{source}-- {relationship} -->{target}\n"

                    for link in data["links"]:
                        # check if link["target"] is array
                        relationship = (
                            link["relationship"] if "relationship" in link else None
                        )
                        if isinstance(link["target"], list):
                            for target in link["target"]:
                                mermaid += make_relationship(
                                    link["source"], target, relationship
                                )
                        else:
                            mermaid += make_relationship(
                                link["source"], link["target"], relationship
                            )

                    # write response to markdown file
                    with open(f"output/{term1}_{term2}.md", "w") as f:
                        for response in reponses:
                            f.write("### Response\n")
                            f.write(response)
                            f.write("\n\n")
                        f.write("```mermaid\n")
                        f.write(mermaid)
                        f.write(
                            "classDef sources fill:#034591,stroke:#000,stroke-width:2px;\n"
                        )
                        f.write("class 0 sources\n")
                        f.write("class 1 sources\n")
                        f.write("```")
                except Exception as e:
                    print("Error parsing json")
                    print(response)
                    pass


        break


if __name__ == "__main__":
    main()
