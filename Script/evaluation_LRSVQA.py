import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
import requests
from io import BytesIO
import re
import torch.nn.functional as F
import copy
from collections import defaultdict
Image.MAX_IMAGE_PIXELS = 10000000000

import nltk
from nltk.corpus import wordnet as wn

nltk.data.path.append('/nltk_data-gh-pages/nltk_data')

def are_synonyms(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    for synset1 in synsets1:
        for synset2 in synsets2:
            if synset1.path_similarity(synset2) is not None and synset1.path_similarity(synset2) > 0.8:
                return True
    return False

def evaluate_dataset(base):

    all_categories = set(item['category'] for item in base)
    category_correct = {category: 0 for category in all_categories}
    category_total = {category: 0 for category in all_categories}

    total_correct = 0
    total_samples = len(base)

    print("Processing evaluations...")
    for item in tqdm(base):
        gt = item.get('ground_truth', '').lower()
        answer = item.get('text', '').lower()
        category = item.get('category')

        if not all([gt, category]):
            print(f"Skipping item due to missing 'ground_truth' or 'category': {item}")
            total_samples -=1  
            continue

        is_correct = False
        if gt == answer:
            is_correct = True
        elif are_synonyms(gt, answer):
            print(f"Synonym match (counted as correct): ground_truth='{gt}', answer='{answer}'")
            is_correct = True

        if is_correct:
            total_correct += 1
            category_correct[category] += 1
        
        category_total[category] += 1

    print("\n--- Evaluation Results ---")
    print(f'Total Correct: {total_correct}')
    print(f'Total Incorrect: {total_samples - total_correct}')
    print(f'Total Samples: {total_samples}')
    print("-" * 25)

    print("Category-wise Accuracies:")
    sorted_categories = sorted(list(all_categories))
    
    category_accuracies = []
    for cat in sorted_categories:
        cat_corr = category_correct[cat]
        cat_total = category_total[cat]
        if cat_total > 0:
            cat_acc = cat_corr / cat_total
            print(f"{cat:<20}: {cat_corr}/{cat_total} ({cat_acc * 100:.2f}%)")
            category_accuracies.append(cat_acc)
        else:
            print(f"{cat:<20}: 0/0 (N/A)")

    print("-" * 25)

    if total_samples > 0:
        overall_acc_oa = total_correct / total_samples
        print(f"Overall Accuracy (OA): {overall_acc_oa * 100:.2f}%")
    else:
        print("Overall Accuracy (OA): N/A (No samples found)")

    if category_accuracies:
        average_acc_aa = sum(category_accuracies) / len(category_accuracies)
        print(f"Average Accuracy (AA): {average_acc_aa * 100:.2f}%")
    else:
        print("Average Accuracy (AA): N/A (No categories with samples found)")


def evaluation_metrics(data_path):
    # 读取数据
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    # 按数据集分组
    datasets = defaultdict(list)
    for item in data:
        dataset_name = item['id'].split('_')[0]
        datasets[dataset_name].append(item)
    
    # 对每个数据集分别评估
    for dataset_name, dataset_data in datasets.items():
        print(f"\n{'='*50}")
        print(f"Evaluating dataset: {dataset_name}")
        print('='*50)
        evaluate_dataset(dataset_data)

def eval_model(args):
    answers_file = args.results_file
    evaluation_metrics(answers_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="./answers.jsonl")
    args = parser.parse_args()
    
    eval_model(args)
