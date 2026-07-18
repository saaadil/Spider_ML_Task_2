import json
from pathlib import Path
from config import CONFIG, SOURCES, EMERGENCY_KEYWORDS, UNSAFE_PATTERNS

def format_response(result: dict) -> str:  # format dict to string
    lines = [  # initialize lines
        f"Question : {result.get('question', 'N/A')}",  # add question
        f"Safe : {'Yes' if result.get('is_safe', True) else 'No - ' + result.get('safety_type', '')}",  # add safety status
        f"Answer : {result.get('answer', '')}",  # add main answer
        f"Confidence : {result.get('confidence', 0):.4f} ({result.get('confidence_level', 'N/A')})",  # add score
        f"Sources ({len(result.get('sources', []))} retrieved):"  # add source count
    ]
    for src in result.get("sources", []):  # loop references
        lines.append(f"  [{src['id']}] {src['source']}" + (f" | {src['focus']}" if src.get('focus') else "") + f" (relevance={src['relevance']:.3f})")  # format ref
        lines.append(f"      {src['excerpt'][:120]}...")  # format excerpt
    return "\n".join(lines)  # join all strings

def print_response(result: dict):  # print to terminal
    print("\n" + format_response(result) + "\n")  # output block

def save_example_outputs(results: list, path: Path):  # save raw examples
    with open(path, "w", encoding="utf-8") as f:  # open text file
        f.write("EXAMPLE QUERIES AND OUTPUTS\nHealthcare Information Assistant\n\n")  # write header
        for i, r in enumerate(results, 1):  # iterate outputs
            f.write(f"Example {i}\n{format_response(r)}\n\n")  # write content
    print(f"  Example outputs -> {path}")  # print log status
