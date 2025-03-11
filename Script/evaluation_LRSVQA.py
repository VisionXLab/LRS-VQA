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

def evaluate_dataset(dataset_data):
    all_categories = set(item['category'] for item in dataset_data)
    category_correct = {category: 0 for category in all_categories}
    category_incorrect = {category: 0 for category in all_categories}
    correct = 0
    incorrect = 0
    
    for item in dataset_data:
        gt = item['gt_answer'].lower()
        # 处理answer为None的情况
        answer = item['answer']
        if answer is None:
            answer = ""  # 将None转换为空字符串
        else:
            answer = answer.lower().rstrip('.')  # 移除答案末尾的句点
            
        category = item['category'].lower()
        
        if gt == answer:
            correct += 1
            category_correct[category] += 1
        else:
            if answer and are_synonyms(gt, answer):  # 确保answer不为空再检查同义词
                print(f'synonyms:{gt} and {answer}')
                correct += 1
                category_correct[category] += 1
            else:
                incorrect += 1
                category_incorrect[category] += 1

    # 打印结果
    print(f'Correct: {correct}')
    print(f'Incorrect: {incorrect}')
    print(f'Total: {correct + incorrect}')
    
    # 计算每个类别的精度
    percentage_list = []
    print("\nCategory-wise accuracies:")
    sorted_categories = sorted(category_correct.keys())
    
    overall_correct = 0
    overall_total = 0
    
    for cat in sorted_categories:
        cat_corr = category_correct[cat]
        cat_total = cat_corr + category_incorrect[cat]
        cat_acc = cat_corr / cat_total if cat_total > 0 else 0
        print(f"{cat}: {cat_corr}/{cat_total} ({cat_acc * 100:.2f}%)")
        percentage_list.append(cat_acc * 100)
        overall_correct += cat_corr
        overall_total += cat_total
    
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0
    print(f"Overall Acc: {overall_acc*100:.2f}%")
    percentage_list.append(overall_acc * 100)
    
    # 打印纯数字的百分比值
    print("\nPercentages:")
    for p in percentage_list:
        s = f"{p:.2f}".rstrip('0').rstrip('.')
        print(s)

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
