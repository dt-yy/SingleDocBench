import os
import re
import pdb
import json
import shutil
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
import statistics

# 添加Levenshtein距离计算函数
def levenshtein_distance(s1, s2):
    """
    计算两个字符串之间的Levenshtein编辑距离
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def normalized_edit_distance(s1, s2):
    """
    计算归一化的编辑距离，返回0到1之间的值
    0表示完全相同，1表示完全不同
    """
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 1.0
    
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return distance / max_len

# 之前的辅助函数保持不变
def replace_dots(latex):
    pattern = r'\.{3,}'
    latex = re.sub(pattern, r'\\dots', latex)
    pattern = r'\\ldots'
    latex = re.sub(pattern, r'\\dots', latex)
    return latex

def clean_error_macro(latex):
    pattern = r'\\differentialD'
    latex = re.sub(pattern, "d", latex)
    pattern = r'\\mathrel{=\\mkern-4mu }'
    latex = re.sub(pattern, "=", latex)
    pattern = r'\\space'
    latex = re.sub(pattern, "", latex)
    pattern = r'\\overline{}'
    latex = re.sub(pattern, "", latex)
    pattern = r'\\lrArr'
    latex = re.sub(pattern, r"\\Leftrightarrow", latex)
    pattern = r'\\bm'
    latex = re.sub(pattern, r"\\boldsymbol", latex)
    pattern = r'\\boldsymbolod'
    latex = re.sub(pattern, r"\\boldsymbol", latex)
    pattern = r'\\normalsize'
    latex = re.sub(pattern, "", latex)
    return latex

def clean_chinese_quotes(latex):
    pattern = r'\^\{\“\}'
    latex = re.sub(pattern, r'^{\\prime\\prime}', latex)
    
    pattern = r'\^\{\”\}'
    latex = re.sub(pattern, r'^{\\prime\\prime}', latex)
    
    pattern = r'\^\{\‘\}'
    latex = re.sub(pattern, r'^{\\prime}', latex)
    
    pattern = r'\^\{\’\}'
    latex = re.sub(pattern, r'^{\\prime}', latex)
    
    pattern = r'\“'
    latex = re.sub(pattern, r'^{\\prime\\prime}', latex)
    
    pattern = r'\”'
    latex = re.sub(pattern, r'^{\\prime\\prime}', latex)
    
    pattern = r'\‘'
    latex = re.sub(pattern, r'^{\\prime}', latex)
    
    pattern = r'\’'
    latex = re.sub(pattern, r'^{\\prime}', latex)
    
    pattern = r'\（'
    latex = re.sub(pattern, r'(', latex)
    
    pattern = r'\）'
    latex = re.sub(pattern, r')', latex)
    
    return latex

def has_consecutive_dots(latex):
    pattern = r'(\\dots){3,}'
    return bool(re.search(pattern, latex))

def clean_super_sub_script(latex):
    pattern = r'\^\{\}'
    latex = re.sub(pattern, "", latex)
    pattern = r'\_\{\}'
    latex = re.sub(pattern, "", latex)
    return latex

def clean_quad(latex):
    latex = re.sub(r'\s+', ' ', latex)
    latex = re.sub(r'(\\quad)+', r'\\quad', latex)
    return latex

import unicodedata

def contains_chinese(text):
    for char in text:
        try:
            if 'CJK' in unicodedata.name(char):
                return True
        except ValueError:
            pass
    return False

from nltk.corpus import wordnet
def replace_text(match):
    word = match.group(0)
    stripped_word = word[6:-1]
    stripped_word = stripped_word.strip()
    
    if len(stripped_word) == 0:
        return ""
    
    if contains_chinese(stripped_word):
        return "\\text" + "{" + stripped_word + "}"
    
    stripped_word = stripped_word.split()[0]
    if stripped_word in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or \
        stripped_word in "abcdefghijklmnopqrstuvwxyz":
        return stripped_word
    
    if len(wordnet.synsets(stripped_word)) > 0:
        return "\\text" + "{" + stripped_word + "}"
    else:
        return stripped_word

def clean_text(latex):
    pattern = r'\\text\{\s+\}'
    latex = re.sub(pattern, "", latex)
    pattern = r'\\textcolor\{[^}]*\}'
    latex = re.sub(pattern, "", latex, flags=re.DOTALL)
    latex = re.sub(r'\\text\{([^}]*)\}', replace_text, latex)
    return latex

def remove_color_tags(latex):
    pattern = r'\\color\{[^}]*\}'
    return re.sub(pattern, '', latex)

def clean_prime(latex):
    pattern = r"''"
    latex = re.sub(pattern, r'^{\\prime\\prime}', latex)
    pattern = r"'"
    latex = re.sub(pattern, r'^{\\prime}', latex)
    pattern = r"\\doubleprime"
    latex = re.sub(pattern, r'\\prime\\prime', latex)
    return latex

def clean_ell(latex):
    latex = re.sub(r"\\mathscr{l}", r'\\ell', latex)
    return latex

def clean_frac(latex):
    latex = re.sub(r"\\dfrac", r'\\frac', latex)
    return latex

def latex_clean(latex):
    latex = clean_super_sub_script(latex)
    latex = replace_dots(latex)
    latex = clean_text(latex)
    latex = clean_error_macro(latex)
    latex = clean_chinese_quotes(latex)
    latex = clean_quad(latex)
    latex = remove_color_tags(latex)
    latex = clean_prime(latex)
    latex = clean_ell(latex)
    latex = clean_frac(latex)
    latex = clean_super_sub_script(latex)
    return latex

def get_formula(input_path, output_path, formula_gt_dir):
    span_data_final_0826_2k_filtered = list()
    
    # 1. 首先收集所有处理后的数据
    for formula_filr in os.listdir(input_path):
        formula_path = os.path.join(input_path, formula_filr)
        lines = open(formula_path, encoding='utf-8').read().split("\n")
        data_all = [json.loads(line) for line in lines if len(line) > 0]
        print(f"Processing {formula_filr}, found {len(data_all)} samples")
        
        span_data_final_0826_2k = []
        for sample in data_all:
            questionnaire_id = sample["questionnaire_id"]
            data_id = sample["data_id"]
            image_link = sample["prompt"].replace("![image 1]", "")[1:-1]
            
            if sample["evaluation"]["conversation_evaluation"] is None:
                continue
            
            # 无法标注内容
            if any(item["not_ok"] == "true" for item in sample["evaluation"]["conversation_evaluation"]["contents"]):
                continue
            
            if "content" in sample["evaluation"]["conversation_evaluation"]["tag_content"][0]:
                tag_content = sample["evaluation"]["conversation_evaluation"]["tag_content"][0]["content"]
            else:
                tag_content = None
            
            span_data_final_0826_2k.append({
                "questionnaire_id": questionnaire_id,
                "data_id": data_id,
                "image_link": image_link,
                "latex": sample["evaluation"]["conversation_evaluation"]["contents"][0]["content"],
                "tag": tag_content,
            })
        
        for sample in span_data_final_0826_2k:
            if "\\textbf" in sample["latex"] or \
                "\\textit" in sample["latex"] or \
                "《" in sample["latex"] or "》" in sample["latex"] or \
                "\\columneqq" in sample["latex"] or "\\phantom" in sample["latex"] or \
                "\\placeholder" in sample["latex"]:
                continue
                
            latex = sample["latex"]
            matches = re.findall(r'\\text\{([^}]*)\}', latex)
            if any(
                "“" in m or "”" in m or "‘" in m or "’" in m or "\\" in m or \
                "'" in m or '"' in m or "''" in m or '""' in m or "-" in m or "&" in m \
                or len(m.strip()) == 0 for m in matches
            ):
                continue
                
            if sample["tag"] and "\\" in sample["tag"]:
                continue
                
            latex_cleaned = latex_clean(latex)
            sample["latex"] = latex_cleaned
            span_data_final_0826_2k_filtered.append(sample)
    
    # 2. 处理LaTeX格式并添加tag
    for sample_idx, sample in enumerate(span_data_final_0826_2k_filtered):
        latex = sample["latex"].strip()
        
        # 移除LaTeX公式的标记符号
        if latex.startswith("$$"):
            latex = latex[2:]
            latex = latex.strip()
        if latex.startswith("$"):
            latex = latex[1:]
            latex = latex.strip()
        if latex.startswith("\\["):
            latex = latex[2:]
            latex = latex.strip()
        if latex.endswith("$$"):
            latex = latex[:-2]
            latex = latex.strip()
        if latex.endswith("$"):
            latex = latex[:-1]
            latex = latex.strip()
        if latex.endswith("\\]"):
            latex = latex[:-2]
            latex = latex.strip()
        
        span_data_final_0826_2k_filtered[sample_idx]["latex"] = latex
        
        if sample["tag"]:
            span_data_final_0826_2k_filtered[sample_idx]["latex"] += " \\tag" + "{" + sample["tag"] + "}"
    
    # 初始化统计变量
    edit_distances = []
    normalized_distances = []
    
    # 3. 保存JSON文件并关联对应的.md文件，同时计算编辑距离
    for sample in tqdm(span_data_final_0826_2k_filtered, desc="Processing samples"):
        image_link = sample["image_link"]
        
        # 从图片链接中提取block_name
        block_name = os.path.basename(image_link)
        
        # 移除图片扩展名，获取基础文件名
        base_name = os.path.splitext(block_name)[0]  # 例如：block-PI7rE7OUaF-BED-0
        
        # 获取处理后的latex
        processed_latex = sample["latex"]
        
        # 初始化formula_gt和编辑距离
        formula_gt_content = ""
        edit_distance = None
        normalized_dist = None
        
        # 检查对应的.md文件是否存在
        md_file_path = os.path.join(formula_gt_dir, f"{base_name}.md")
        
        if os.path.exists(md_file_path):
            try:
                with open(md_file_path, "r", encoding="utf-8") as f:
                    formula_gt_content = f.read().strip()
                
                # 计算编辑距离
                if formula_gt_content:
                    edit_distance = levenshtein_distance(processed_latex, formula_gt_content)
                    normalized_dist = normalized_edit_distance(processed_latex, formula_gt_content)
                    
                    # 添加到统计列表中
                    edit_distances.append(edit_distance)
                    normalized_distances.append(normalized_dist)
                    
            except Exception as e:
                print(f"Error reading {md_file_path}: {e}")
                formula_gt_content = ""
        else:
            print(f"Warning: No .md file found for {base_name}")
        
        # 更新sample数据，添加formula_gt和编辑距离
        sample["formula_gt"] = formula_gt_content
        if edit_distance is not None:
            sample["edit_distance"] = edit_distance
            sample["normalized_edit_distance"] = normalized_dist
        else:
            sample["edit_distance"] = -1  # 表示没有可比较的内容
            sample["normalized_edit_distance"] = -1
        
        # 添加block_name到输出
        sample["block_name"] = base_name
        
        # 保存JSON文件
        json_file_path = os.path.join(output_path, f"{base_name}.json")
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
    
    # 4. 计算并输出编辑距离的平均分
    if edit_distances:
        avg_edit_distance = sum(edit_distances) / len(edit_distances)
        avg_normalized_distance = sum(normalized_distances) / len(normalized_distances)
        
        # 计算更多统计信息
        min_edit_distance = min(edit_distances)
        max_edit_distance = max(edit_distances)
        median_edit_distance = statistics.median(edit_distances) if len(edit_distances) >= 1 else 0
        
        min_norm_distance = min(normalized_distances)
        max_norm_distance = max(normalized_distances)
        median_norm_distance = statistics.median(normalized_distances) if len(normalized_distances) >= 1 else 0
        
        print("\n" + "="*60)
        print("编辑距离统计结果:")
        print("="*60)
        print(f"有效比较样本数量: {len(edit_distances)}")
        print(f"原始编辑距离统计:")
        print(f"  平均值: {avg_edit_distance:.2f}")
        print(f"  最小值: {min_edit_distance}")
        print(f"  最大值: {max_edit_distance}")
        print(f"  中位数: {median_edit_distance:.2f}")
        print(f"归一化编辑距离统计 (0-1, 0表示完全相同):")
        print(f"  平均值: {avg_normalized_distance:.4f}")
        print(f"  最小值: {min_norm_distance:.4f}")
        print(f"  最大值: {max_norm_distance:.4f}")
        print(f"  中位数: {median_norm_distance:.4f}")
        print("="*60)
        
        # 保存统计结果到文件
        stats_output_path = os.path.join(output_path, "edit_distance_stats.json")
        stats = {
            "total_samples": len(span_data_final_0826_2k_filtered),
            "samples_with_gt": len(edit_distances),
            "edit_distance_stats": {
                "average": avg_edit_distance,
                "min": min_edit_distance,
                "max": max_edit_distance,
                "median": median_edit_distance,
                "all_distances": edit_distances
            },
            "normalized_edit_distance_stats": {
                "average": avg_normalized_distance,
                "min": min_norm_distance,
                "max": max_norm_distance,
                "median": median_norm_distance,
                "all_distances": normalized_distances
            }
        }
        
        with open(stats_output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"统计结果已保存到: {stats_output_path}")
    else:
        print("\n警告: 没有找到任何有效的编辑距离数据（可能没有对应的.md文件）")

if __name__ == "__main__":
    input_path = r"D:\pdf-bench-v2\SingleDocBench\formula_data"
    output_path = r"D:\pdf-bench-v2\SingleDocBench\formula_result"
    formula_gt_dir = r"D:\pdf-bench-v2\SingleDocBench\formula_gt\formula_gt"
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 检查nltk数据是否可用
    try:
        from nltk.corpus import wordnet
    except LookupError:
        print("Downloading nltk wordnet data...")
        import nltk
        nltk.download('wordnet')
    
    get_formula(input_path, output_path, formula_gt_dir)
    print("Processing completed!")