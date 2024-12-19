import json
from difflib import SequenceMatcher
from sympy import simplify, sympify
import os

def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def similarity_score(master_eq, pipeline_eq):
    # For string similarity
    return SequenceMatcher(None, master_eq, pipeline_eq).ratio()


def benchmark_pipeline(master_data, pipeline_data):
    total_exact_score = 0
    total_similarity_score = 0
    total_math_equiv_score = 0
    count = len(master_data)

    for image_name, master_eq in master_data.items():
        pipeline_eq = pipeline_data.get(image_name, "")
        total_similarity_score += similarity_score(master_eq, pipeline_eq)

    return {
        "Similarity Score": total_similarity_score / count,
        "Mathematical Equivalence Score": total_math_equiv_score / count
    }