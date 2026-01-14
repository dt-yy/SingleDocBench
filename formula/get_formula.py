import os
import re
import json
import time
from tqdm.auto import tqdm
from collections import defaultdict

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

# 辅助函数保持不变
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
    pattern = r'\^\{\"\"\}'
    latex = re.sub(pattern, r'^{\\prime\\prime}', latex)
    
    pattern = r'\^\{\"\}'
    latex = re.sub(pattern, r'^{\\prime}', latex)
    
    pattern = r'\"'
    latex = re.sub(pattern, r'^{\\prime\\prime}', latex)
    
    pattern = r"''"
    latex = re.sub(pattern, r'^{\\prime\\prime}', latex)
    
    pattern = r"'"
    latex = re.sub(pattern, r'^{\\prime}', latex)
    
    pattern = r'\\doubleprime'
    latex = re.sub(pattern, r'\\prime\\prime', latex)
    
    return latex

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

def sanitize_filename(filename):
    """
    清理文件名，移除空格和特殊字符
    """
    # 移除空格
    filename = filename.replace(" ", "_")
    
    # 移除其他可能导致问题的字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # 移除首尾的点
    filename = filename.strip('.')
    
    return filename

def process_single_jsonl(jsonl_file_path, jsonl_filename, formula_gt_dir):
    """
    处理单个JSONL文件，只计算平均值
    
    Args:
        jsonl_file_path (str): JSONL文件完整路径
        jsonl_filename (str): JSONL文件名（不带扩展名）
        formula_gt_dir (str): GT文件目录
        
    Returns:
        tuple: (原始编辑距离平均值, 归一化编辑距离平均值, 总样本数, 有GT的样本数)
    """
    print(f"处理文件: {jsonl_filename}")
    
    # 读取JSONL文件
    try:
        with open(jsonl_file_path, encoding='utf-8') as f:
            lines = f.read().split("\n")
    except Exception as e:
        print(f"  错误: 无法读取文件 {jsonl_filename}: {e}")
        return 0.0, 0.0, 0, 0
    
    data_all = [json.loads(line) for line in lines if len(line) > 0]
    print(f"  读取到 {len(data_all)} 个样本")
    
    # 处理每个样本
    samples = []
    for sample in data_all:
        # 检查数据有效性
        if sample.get("evaluation", {}).get("conversation_evaluation") is None:
            continue
        
        # 无法标注内容
        if any(item.get("not_ok") == "true" for item in sample["evaluation"]["conversation_evaluation"].get("contents", [])):
            continue
        
        # 获取图片链接
        try:
            image_link = sample["prompt"].replace("![image 1]", "")[1:-1]
        except Exception:
            continue
        
        # 获取标签内容
        tag_content = None
        if "tag_content" in sample["evaluation"]["conversation_evaluation"] and sample["evaluation"]["conversation_evaluation"]["tag_content"]:
            if "content" in sample["evaluation"]["conversation_evaluation"]["tag_content"][0]:
                tag_content = sample["evaluation"]["conversation_evaluation"]["tag_content"][0]["content"]
        
        # 获取LaTeX内容
        latex_content = ""
        if "contents" in sample["evaluation"]["conversation_evaluation"] and sample["evaluation"]["conversation_evaluation"]["contents"]:
            latex_content = sample["evaluation"]["conversation_evaluation"]["contents"][0].get("content", "")
        
        samples.append({
            "image_link": image_link,
            "latex": latex_content,
            "tag": tag_content,
        })
    
    # 过滤样本
    filtered_samples = []
    for sample in samples:
        latex = sample["latex"]
        
        # 跳过包含特定内容的样本
        if any(bad in latex for bad in ["\\textbf", "\\textit", "《", "》", "\\columneqq", "\\phantom", "\\placeholder"]):
            continue
        
        # 检查文本内容
        matches = re.findall(r'\\text\{([^}]*)\}', latex)
        if any(
            any(ch in m for ch in ["\"", "'", "''", '""', "-", "&"]) or \
            len(m.strip()) == 0 for m in matches
        ):
            continue
        
        # 检查标签
        if sample["tag"] and "\\" in sample["tag"]:
            continue
        
        # 清理LaTeX
        latex_cleaned = latex_clean(latex)
        sample["latex"] = latex_cleaned
        filtered_samples.append(sample)
    
    print(f"  过滤后剩余 {len(filtered_samples)} 个样本")
    
    # 处理LaTeX格式并添加tag
    for sample in filtered_samples:
        latex = sample["latex"].strip()
        
        # 移除LaTeX公式的标记符号
        markers = [("$$", 2), ("$", 1), ("\\[", 2)]
        for start_marker, length in markers:
            if latex.startswith(start_marker):
                latex = latex[length:].strip()
        
        markers_end = [("$$", 2), ("$", 1), ("\\]", 2)]
        for end_marker, length in markers_end:
            if latex.endswith(end_marker):
                latex = latex[:-length].strip()
        
        # 添加标签
        if sample["tag"]:
            latex += " \\tag" + "{" + sample["tag"] + "}"
        
        sample["latex"] = latex
    
    # 初始化统计变量
    edit_distances_sum = 0
    normalized_distances_sum = 0
    valid_samples_count = 0
    
    # 计算编辑距离
    for sample in tqdm(filtered_samples, desc=f"计算 {jsonl_filename}", leave=False):
        # 获取处理后的latex
        processed_latex = sample["latex"]
        
        # 从图片链接中提取block_name
        block_name = os.path.basename(sample["image_link"])
        base_name = os.path.splitext(block_name)[0]
        
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
                    
                    # 累加距离
                    edit_distances_sum += edit_distance
                    normalized_distances_sum += normalized_dist
                    valid_samples_count += 1
                    
            except Exception:
                # 忽略读取错误
                pass
    
    # 计算平均值
    avg_edit_distance = edit_distances_sum / valid_samples_count if valid_samples_count > 0 else 0.0
    avg_normalized_distance = normalized_distances_sum / valid_samples_count if valid_samples_count > 0 else 0.0
    
    print(f"  有效样本: {valid_samples_count}/{len(filtered_samples)}")
    print(f"  原始编辑距离平均值: {avg_edit_distance:.2f}")
    print(f"  归一化编辑距离平均值: {avg_normalized_distance:.4f}")
    
    return avg_edit_distance, avg_normalized_distance, len(filtered_samples), valid_samples_count

def get_formula(input_path, formula_gt_dir):
    """
    处理所有JSONL文件，只计算平均值
    
    Args:
        input_path (str): 输入目录，包含多个JSONL文件
        formula_gt_dir (str): GT文件目录
    """
    # 获取所有JSONL文件
    jsonl_files = []
    for item in os.listdir(input_path):
        item_path = os.path.join(input_path, item)
        if os.path.isfile(item_path) and item.endswith('.jsonl'):
            jsonl_files.append((item_path, item))
    
    print(f"找到 {len(jsonl_files)} 个JSONL文件")
    print("=" * 60)
    
    if not jsonl_files:
        print("错误: 没有找到JSONL文件")
        return
    
    # 处理每个JSONL文件
    results = []
    total_edit_distance_sum = 0
    total_normalized_distance_sum = 0
    total_valid_samples = 0
    
    for jsonl_file_path, jsonl_filename in jsonl_files:
        # 提取文件名（不带扩展名）
        jsonl_name = os.path.splitext(jsonl_filename)[0]
        
        # 处理单个JSONL文件
        avg_edit, avg_norm, total_samples, valid_samples = process_single_jsonl(
            jsonl_file_path, jsonl_name, formula_gt_dir
        )
        
        results.append({
            "file": jsonl_name,
            "avg_edit_distance": avg_edit,
            "avg_normalized_edit_distance": avg_norm,
            "total_samples": total_samples,
            "valid_samples": valid_samples
        })
        
        # 累加总体统计
        total_edit_distance_sum += avg_edit * valid_samples
        total_normalized_distance_sum += avg_norm * valid_samples
        total_valid_samples += valid_samples
        
        print()
    
    print("=" * 60)
    print("各JSONL文件编辑距离平均值统计:")
    print("=" * 60)
    
    # 输出每个JSONL文件的统计结果
    for result in results:
        print(f"文件: {result['file']}")
        print(f"  总样本数: {result['total_samples']}")
        print(f"  有效样本数: {result['valid_samples']}")
        print(f"  原始编辑距离平均值: {result['avg_edit_distance']:.2f}")
        print(f"  归一化编辑距离平均值: {result['avg_normalized_edit_distance']:.4f}")
        print()
    
    

if __name__ == "__main__":
    input_path = r"D:\pdf-bench-v2\SingleDocBench\formula_data"
    formula_gt_dir = r"D:\pdf-bench-v2\SingleDocBench\formula_gt\formula_gt"
    
    # 检查nltk数据是否可用
    try:
        from nltk.corpus import wordnet
    except LookupError:
        print("下载nltk wordnet数据...")
        import nltk
        nltk.download('wordnet')
    
    start_time = time.time()
    get_formula(input_path, formula_gt_dir)
    
    end_time = time.time()
    print(f"\n处理完成! 总耗时: {end_time - start_time:.2f} 秒")