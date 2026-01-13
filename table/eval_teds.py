import os
import re
import sys
import json
import time
import argparse
from tqdm import tqdm
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from table_metrics import TEDS
from otsl2html import otsl_to_html
from table_html_norm import normalized_html_table


def extract_table(html):
    l_pattern, r_pattern = re.escape("<table"), re.escape("</table>")
    pattern = re.compile(r'{}.*?{}'.format(l_pattern, r_pattern), re.DOTALL)
    table_htmls = pattern.findall(html)
    return table_htmls if table_htmls else [html]

def formula_post_process(text):
    pattern = r'\\\(([^)]*?)\\\)'
    def replace_func(match):
        content = match.group(1).strip()
        return f'${content}$'
    result = re.sub(pattern, replace_func, text)
    return result

def remove_spaces_in_td(html_text):
    """
    去除所有<td>标签内容中的空格
    """
    # <td[^>]*> 匹配开始标签（可能带有属性; (.*?) 非贪婪匹配标签内容; </td> 匹配结束标签
    pattern = r'(<td[^>]*>)(.*?)(</td>)'
    def replace_func(match):
        # 获取开始标签、内容和结束标签
        start_tag = match.group(1)  # <td> 或 <td ...>
        content = match.group(2)    # 标签内容
        end_tag = match.group(3)    # </td>
        # 去掉内容中的所有空格
        # content_no_spaces = content.replace(' ', '')
        content_no_spaces = re.sub(r'\s+', '', content)
        # 返回处理后的完整标签
        return start_tag + content_no_spaces + end_tag
    # 使用 re.DOTALL 标志使 . 匹配换行符
    result = re.sub(pattern, replace_func, html_text, flags=re.DOTALL)
    return result

def norm_text(text):
    text = formula_post_process(text)
    text = text.replace("✓", "√").replace("✔", "√").replace("\checkmark", "√").\
                replace("α", r"\alpha").replace("β", r"\beta").replace("γ", r"\gama").replace("μ", r"\mu").\
                replace("±", "\pm").replace("Ø", "∅").replace("-", "—").\
                replace("$", "")
    text = remove_spaces_in_td(text)
    return text


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--gt-dir", '-g', type=str, default='benchmark')
    #parser.add_argument("--pred-dir", '-p', type=str, default="results/test")
    #args = parser.parse_args()
    #args.gt_dir = args.gt_dir.rstrip("/")
    #args.pred_dir = args.pred_dir.rstrip("/")
    gt_dir = r"D:\pdf-bench-v2\SingleDocBench\table_gt\table_gt"
    pred_dir = r"D:\pdf-bench-v2\SingleDocBench\result\labels"

    start_time = time.time()
    norm_func = normalized_html_table
    print("=> gt nums:", len(os.listdir(gt_dir)), "pred nums:", len(os.listdir(pred_dir)))
    _, pfix = os.path.splitext(os.listdir(pred_dir)[0])

    gt_htmls, pred_htmls, gt_fnames = [], [], []
    for fname in tqdm(os.listdir(gt_dir)):
        basename, _ = os.path.splitext(fname)
        if os.path.exists(os.path.join(pred_dir, basename + pfix)):
            pred_html = open(os.path.join(pred_dir, basename + pfix), 'r', encoding='utf-8').read()
            pred_html = extract_table(pred_html)[0]
            pred_html = otsl_to_html(pred_html)
            pred_html = norm_text(pred_html)
            gt_html = open(os.path.join(gt_dir, fname), 'r', encoding='utf-8').read()
            gt_html_list = extract_table(gt_html)
            
            for gt_html in gt_html_list:
                gt_html = otsl_to_html(gt_html)
                gt_html = norm_text(gt_html)
                try:
                    gt_html = norm_func(gt_html)
                except Exception as e:
                    print(f"{fname} gt normlize error:", e)
                try:
                    pred_html = norm_func(pred_html)
                except Exception as e:
                    print(f"{fname} pred normlize error:", e)
                
                gt_html = "<html><body>" + gt_html[0:50000] + "</body></html>"
                pred_html = "<html><body>" + pred_html[0:50000] + "</body></html>"
                
                pred_htmls.append(pred_html)
                gt_htmls.append(gt_html)
                gt_fnames.append(basename)
        else:
            print(f"{fname} not found in pred_dir")
            continue
    
    print("loaded gt and pred htmls:", len(gt_htmls), len(pred_htmls))
    # compute teds
    n_jobs = 128
    teds = TEDS(n_jobs=n_jobs, ignore_nodes='b')
    scores1 = teds.batch_evaluate_html(pred_htmls, gt_htmls)
    
    n_jobs = 128
    teds = TEDS(n_jobs=n_jobs, ignore_nodes='b', structure_only=True)
    scores2 = teds.batch_evaluate_html(pred_htmls, gt_htmls)
    
    teds_scores = {}
    for fname, teds, steds in zip(gt_fnames, scores1, scores2):
        if fname not in teds_scores:
            teds_scores[fname] = {"TEDS": teds, "TEDS-S": steds}
        else:
            old_score = teds_scores[fname]
            teds_scores[fname] = {
                "TEDS": max(old_score["TEDS"], teds), "TEDS-S": max(old_score["TEDS-S"], steds)
            }
    print("teds_scores:", teds_scores)
    