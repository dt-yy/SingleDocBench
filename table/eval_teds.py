import os
import re
import sys
import json
import time
import argparse
from tqdm import tqdm
from collections import defaultdict

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from table_metrics import TEDS
from otsl2html import otsl_to_html
from table_html_norm import normalized_html_table


def extract_table(html):
    """提取HTML中的表格内容"""
    l_pattern, r_pattern = re.escape("<table"), re.escape("</table>")
    pattern = re.compile(r'{}.*?{}'.format(l_pattern, r_pattern), re.DOTALL)
    table_htmls = pattern.findall(html)
    return table_htmls if table_htmls else [html]


def formula_post_process(text):
    """公式后处理，将LaTeX格式标准化"""
    pattern = r'\\\(([^)]*?)\\\)'
    
    def replace_func(match):
        content = match.group(1).strip()
        return f'${content}$'
    
    result = re.sub(pattern, replace_func, text)
    return result


def remove_spaces_in_td(html_text):
    """去除所有<td>标签内容中的空格"""
    pattern = r'(<td[^>]*>)(.*?)(</td>)'
    
    def replace_func(match):
        start_tag = match.group(1)  # <td> 或 <td ...>
        content = match.group(2)    # 标签内容
        end_tag = match.group(3)    # </td>
        content_no_spaces = re.sub(r'\s+', '', content)
        return start_tag + content_no_spaces + end_tag
    
    result = re.sub(pattern, replace_func, html_text, flags=re.DOTALL)
    return result


def norm_text(text):
    """文本标准化处理"""
    text = formula_post_process(text)
    text = text.replace("✓", "√").replace("✔", "√").replace("\checkmark", "√").\
                replace("α", r"\alpha").replace("β", r"\beta").replace("γ", r"\gama").replace("μ", r"\mu").\
                replace("±", "\pm").replace("Ø", "∅").replace("-", "—").\
                replace("$", "")
    text = remove_spaces_in_td(text)
    return text


def calculate_average_scores(teds_scores):
    """计算平均TEDS分数"""
    if not teds_scores:
        return 0.0, 0.0
    
    total_teds = 0.0
    total_steds = 0.0
    count = len(teds_scores)
    
    for score in teds_scores.values():
        total_teds += score.get("TEDS", 0.0)
        total_steds += score.get("TEDS-S", 0.0)
    
    avg_teds = total_teds / count if count > 0 else 0.0
    avg_steds = total_steds / count if count > 0 else 0.0
    
    return avg_teds, avg_steds


def process_folder(gt_dir, pred_folder_path, folder_name):
    """
    处理单个预测文件夹，计算该文件夹的TEDS分数
    
    Args:
        gt_dir (str): GT文件目录
        pred_folder_path (str): 预测文件夹路径
        folder_name (str): 文件夹名称
        
    Returns:
        dict: 包含该文件夹的分数统计信息
    """
    # 获取labels文件夹路径
    labels_dir = os.path.join(pred_folder_path, "labels")
    
    if not os.path.exists(labels_dir):
        print(f"  警告: {folder_name} 中没有找到labels文件夹")
        return None
    
    print(f"  处理文件夹: {folder_name}")
    print(f"    预测文件数量: {len(os.listdir(labels_dir))}")
    
    # 获取预测文件扩展名
    pred_files = os.listdir(labels_dir)
    if not pred_files:
        print(f"  警告: {folder_name} 的labels文件夹为空")
        return None
    
    _, pfix = os.path.splitext(pred_files[0])
    gt_htmls, pred_htmls, gt_fnames = [], [], []
    
    # 统计匹配的文件数
    matched_count = 0
    
    # 读取并处理GT和预测文件
    for fname in tqdm(os.listdir(gt_dir), desc=f"处理 {folder_name}", leave=False):
        basename, _ = os.path.splitext(fname)
        pred_path = os.path.join(labels_dir, basename + pfix)
        
        if os.path.exists(pred_path):
            matched_count += 1
            # 读取和处理预测文件
            with open(pred_path, 'r', encoding='utf-8') as f:
                pred_html = f.read()
            pred_html = extract_table(pred_html)[0]
            pred_html = otsl_to_html(pred_html)
            pred_html = norm_text(pred_html)
            
            # 读取和处理GT文件
            with open(os.path.join(gt_dir, fname), 'r', encoding='utf-8') as f:
                gt_html = f.read()
            gt_html_list = extract_table(gt_html)
            
            for gt_html in gt_html_list:
                gt_html = otsl_to_html(gt_html)
                gt_html = norm_text(gt_html)
                
                try:
                    gt_html = normalized_html_table(gt_html)
                except Exception as e:
                    print(f"    {fname} gt normalize error:", e)
                
                try:
                    pred_html = normalized_html_table(pred_html)
                except Exception as e:
                    print(f"    {fname} pred normalize error:", e)
                
                # 添加HTML包装并限制长度
                gt_html = "<html><body>" + gt_html[0:50000] + "</body></html>"
                pred_html = "<html><body>" + pred_html[0:50000] + "</body></html>"
                
                pred_htmls.append(pred_html)
                gt_htmls.append(gt_html)
                gt_fnames.append(basename)
    
    print(f"    匹配的预测文件: {matched_count}/{len(os.listdir(gt_dir))}")
    
    if not gt_htmls:
        print(f"  警告: {folder_name} 没有匹配的预测文件")
        return None
    
    # 计算TEDS分数
    n_jobs = 128
    
    # 计算完整TEDS
    teds = TEDS(n_jobs=n_jobs, ignore_nodes='b')
    scores1 = teds.batch_evaluate_html(pred_htmls, gt_htmls)
    
    # 计算结构TEDS-S
    teds = TEDS(n_jobs=n_jobs, ignore_nodes='b', structure_only=True)
    scores2 = teds.batch_evaluate_html(pred_htmls, gt_htmls)
    
    # 整理分数，每个文件取多个表格中的最高分
    teds_scores = {}
    for fname, teds, steds in zip(gt_fnames, scores1, scores2):
        if fname not in teds_scores:
            teds_scores[fname] = {"TEDS": teds, "TEDS-S": steds}
        else:
            old_score = teds_scores[fname]
            teds_scores[fname] = {
                "TEDS": max(old_score["TEDS"], teds), 
                "TEDS-S": max(old_score["TEDS-S"], steds)
            }
    
    # 计算平均分数
    avg_teds, avg_steds = calculate_average_scores(teds_scores)
    
    return {
        "folder_name": folder_name,
        "avg_teds": avg_teds,
        "avg_steds": avg_steds,
        "file_count": len(teds_scores),
        "matched_count": matched_count,
        "individual_scores": teds_scores
    }


def main():
    gt_dir = r"D:\pdf-bench-v2\SingleDocBench\table_gt\table_gt"
    pred_base_dir = r"D:\pdf-bench-v2\SingleDocBench\table_result"
    
    start_time = time.time()
    print("=" * 60)
    print("开始处理多个预测文件夹...")
    print(f"GT目录: {gt_dir}")
    print(f"预测基础目录: {pred_base_dir}")
    print("=" * 60)
    
    # 获取GT文件数量
    gt_files = os.listdir(gt_dir)
    print(f"GT文件总数: {len(gt_files)}")
    
    # 获取所有预测文件夹
    pred_folders = []
    for item in os.listdir(pred_base_dir):
        item_path = os.path.join(pred_base_dir, item)
        if os.path.isdir(item_path):
            pred_folders.append(item)
    
    print(f"找到 {len(pred_folders)} 个预测文件夹")
    print()
    
    if not pred_folders:
        print("错误: 没有找到预测文件夹")
        return
    
    # 处理每个文件夹
    folder_results = []
    all_individual_scores = defaultdict(list)  # 按文件名分组存储所有文件夹的分数
    
    for folder_name in pred_folders:
        pred_folder_path = os.path.join(pred_base_dir, folder_name)
        result = process_folder(gt_dir, pred_folder_path, folder_name)
        
        if result:
            folder_results.append(result)
            
            # 收集所有文件的分数用于计算总平均
            for fname, scores in result["individual_scores"].items():
                all_individual_scores[fname].append({
                    "folder": folder_name,
                    "TEDS": scores["TEDS"],
                    "TEDS-S": scores["TEDS-S"]
                })
    
    print("\n" + "=" * 60)
    print("各文件夹TEDS分数统计:")
    print("=" * 60)
    
    # 输出每个文件夹的结果
    for result in folder_results:
        print(f"\n文件夹: {result['folder_name']}")
        print(f"  文件数量: {result['file_count']}")
        print(f"  匹配文件: {result['matched_count']}/{len(gt_files)}")
        print(f"  平均TEDS: {result['avg_teds']:.4f}")
        print(f"  平均TEDS-S: {result['avg_steds']:.4f}")
    
    # 计算所有文件的总体平均分数（每个文件在不同文件夹中的最高分）
    print("\n" + "=" * 60)
    print("总体统计（每个文件取各文件夹中的最高分）:")
    print("=" * 60)
    
    # 为每个文件取最高分
    best_scores = {}
    for fname, scores_list in all_individual_scores.items():
        best_teds = max([s["TEDS"] for s in scores_list])
        best_steds = max([s["TEDS-S"] for s in scores_list])
        best_scores[fname] = {
            "TEDS": best_teds,
            "TEDS-S": best_steds,
            "folders": [s["folder"] for s in scores_list]
        }
    
    # 计算总体平均
    if best_scores:
        total_teds = sum([s["TEDS"] for s in best_scores.values()])
        total_steds = sum([s["TEDS-S"] for s in best_scores.values()])
        overall_teds = total_teds / len(best_scores)
        overall_steds = total_steds / len(best_scores)
        
        print(f"\n总文件数: {len(best_scores)}")
        print(f"总体平均TEDS: {overall_teds:.4f}")
        print(f"总体平均TEDS-S: {overall_steds:.4f}")
    else:
        print("没有找到有效的分数数据")
    
    # 保存详细结果到JSON文件
    output_data = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "gt_dir": gt_dir,
        "pred_base_dir": pred_base_dir,
        "gt_file_count": len(gt_files),
        "folder_results": folder_results,
        "overall_scores": {
            "avg_teds": overall_teds if best_scores else 0.0,
            "avg_steds": overall_steds if best_scores else 0.0,
            "total_files": len(best_scores) if best_scores else 0
        },
        "best_scores_per_file": best_scores
    }
    
    # 生成输出文件名
    timestamp_str = time.strftime('%Y%m%d_%H%M%S')
    output_file = f"teds_scores_{timestamp_str}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 保存简化的文本报告
    text_report = f"teds_report_{timestamp_str}.txt"
    with open(text_report, 'w', encoding='utf-8') as f:
        f.write("TEDS分数评估报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GT目录: {gt_dir}\n")
        f.write(f"预测目录: {pred_base_dir}\n")
        f.write(f"GT文件总数: {len(gt_files)}\n\n")
        
        f.write("各文件夹结果:\n")
        f.write("-" * 50 + "\n")
        for result in folder_results:
            f.write(f"文件夹: {result['folder_name']}\n")
            f.write(f"  文件数: {result['file_count']}\n")
            f.write(f"  匹配数: {result['matched_count']}/{len(gt_files)}\n")
            f.write(f"  平均TEDS: {result['avg_teds']:.4f}\n")
            f.write(f"  平均TEDS-S: {result['avg_steds']:.4f}\n\n")
        
        f.write("\n总体结果:\n")
        f.write("-" * 50 + "\n")
        f.write(f"总评估文件数: {len(best_scores)}\n")
        f.write(f"总体平均TEDS: {overall_teds:.4f}\n")
        f.write(f"总体平均TEDS-S: {overall_steds:.4f}\n")
    
    print(f"文本报告已保存到: {text_report}")
    
    elapsed_time = time.time() - start_time
    print(f"\n总耗时: {elapsed_time:.2f}秒")


if __name__ == '__main__':
    main()