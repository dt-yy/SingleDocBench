import os
import re
import json
import argparse
import requests
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any
from html2otsl import html_to_otsl
from table_html_norm import normalized_html_table



def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    读取单个JSONL文件的内容
    
    Args:
        file_path (str): JSONL文件路径
        
    Returns:
        List[Dict[str, Any]]: 文件中的所有JSON对象列表
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        json_obj = json.loads(line)
                        data.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"  警告: 第{line_num}行JSON解析失败: {e}")
                        print(f"  内容: {line[:100]}...")
    except Exception as e:
        print(f"  错误: 读取文件失败: {e}")
    
    return data


def scan_jsonl_files(folder_path: str, recursive: bool = False) -> List[str]:
    """
    扫描文件夹中的JSONL文件
    
    Args:
        folder_path (str): 要扫描的文件夹路径
        recursive (bool): 是否递归扫描子文件夹
        
    Returns:
        List[str]: 找到的JSONL文件路径列表
    """
    jsonl_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return jsonl_files
    
    if not folder.is_dir():
        print(f"错误: '{folder_path}' 不是一个文件夹")
        return jsonl_files
    
    # 扫描文件
    if recursive:
        pattern = "**/*.jsonl"
    else:
        pattern = "*.jsonl"
    
    for file_path in folder.glob(pattern):
        if file_path.is_file():
            jsonl_files.append(str(file_path))
    
    return sorted(jsonl_files)


def load_folder(folder_path: str, recursive: bool = False) -> List[Dict[str, Any]]:
    """
    处理文件夹中的所有JSONL文件
    
    Args:
        folder_path (str): 文件夹路径
        recursive (bool): 是否递归扫描
        
    Returns:
        List[Dict[str, Any]]: 所有JSONL文件的数据合并列表
    """
    print(f"正在扫描文件夹: {folder_path}")
    if recursive:
        print("递归扫描模式: 开启")
    
    # 扫描JSONL文件
    jsonl_files = scan_jsonl_files(folder_path, recursive)
    
    if not jsonl_files:
        print("未找到任何JSONL文件")
        return []
    
    print(f"找到 {len(jsonl_files)} 个JSONL文件")
    
    # 合并所有文件的数据
    all_data = []
    
    # 处理每个文件
    for i, file_path in enumerate(jsonl_files, 1):
        print(f"正在读取 [{i}/{len(jsonl_files)}]: {os.path.basename(file_path)}")
        
        # 读取文件内容
        data = read_jsonl_file(file_path)
        all_data.extend(data)
    
    print(f"读取完成！总共 {len(all_data)} 条数据")
    return all_data

def download_img(img_url: str, save_dir: str):
    img_name = os.path.basename(img_url)
    img_path = os.path.join(save_dir, img_name)
    if not os.path.exists(img_path):
        try:
            res=requests.get(img_url)
            with open(img_path, 'wb') as f:
                f.write(res.content)
        except Exception as e:
            print(f"下载失败: {img_url}")
            return False

    return os.path.exists(img_path)

def write_text(text: str, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)

def extract_img_url(text: str) -> str:
    # 使用正则表达式匹配 ![alt text](url) 格式
    pattern = r'\((https:.*?.png)\)'
    match = re.search(pattern, text)
    
    if match:
        return match.group(1)
    else:
        pattern = r'\((https:.*?.jpg)\)'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return ""

def process_single_data(data: Dict[str, Any]):
    angle_str2int = {
        "rotated_angle_0": 0,
        "rotated_angle_90": 270,
        "rotated_angle_180": 180,
        "rotated_angle_270": 90,
    }
    

    if data['evaluation']['questionnaire_evaluation']['is_invalid_questionnaire']:
        return None, None

    # if data['evaluation']['data_evaluation'] is not None and not all([check['is_pass'] == "yes" for check in data['evaluation']['data_evaluation']]):
    #     return None, None
    table_content = data['evaluation']['conversation_evaluation']
    for item in table_content:
        pred = table_content[item]["boxes"][0]["attributes"]['content']

    try:
        angle = angle_str2int[data['evaluation']['conversation_evaluation']['rotated_angle']]
    except Exception as e:
        angle = 0

    img_url = extract_img_url(data['prompt'])
    try:
        bbox = data['evaluation']['conversation_evaluation'][img_url]['boxes']
        id2bbox = {}
        for box in bbox:
            xmin, ymin, xmax, ymax = box['x'], box['y'], box['x']+box['width'], box['y']+box['height']
            id2bbox[box['id']] = [xmin, ymin, xmax, ymax]
    except Exception as e:
        id2bbox = {}

    gt = {
        "imname": os.path.basename(img_url),
        "angle": angle,
        "html": pred,
        "otsl": html_to_otsl(pred),
        "id2bbox": id2bbox
    }
    return img_url, gt

def process_img_bbox(text, id2bbox, imsize):
    W, H = imsize
    for boxid, bbox in id2bbox.items():
        xmin, ymin, xmax, ymax = bbox
        xmin = str(int(xmin / W * 1000)).zfill(3)
        ymin = str(int(ymin / H * 1000)).zfill(3)
        xmax = str(int(xmax / W * 1000)).zfill(3)
        ymax = str(int(ymax / H * 1000)).zfill(3)
        box_str = f"{xmin} {ymin} {xmax} {ymax}"
        text = text.replace(f'<img src="{boxid}">', '<img src="">')
        # text = text.replace(f'<img src="{boxid}">', f'<img src="<|box_start|>{box_str}<|box_end|>">')
        # text = text.replace(f'<img src="{boxid}">', f'<|ref_start|>image<|ref_end|><|box_start|>{box_str}<|box_end|>')
        # text = text.replace(f'<img src="{boxid}">', f'<|box_start|>{box_str}<|box_end|>')
        # text = text.replace(f'<img src="{boxid}">', f'<|block_start|>{box_str}<|block_end|>')
    return text

def main():
    input_path = r"D:\pdf-bench-v2\SingleDocBench\data"
    output_path = r"D:\pdf-bench-v2\SingleDocBench\result"
    data_list = load_folder(input_path)

    valid_num = 0
    save_dir = os.path.join(output_path, "images")
    os.makedirs(save_dir, exist_ok=True)
    labels_dir = os.path.join(output_path, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    for data in tqdm(data_list):
        img_url, gt = process_single_data(data)
        if img_url and gt is not None:
            flag = download_img(img_url, save_dir)
            if flag:
                img = Image.open(os.path.join(save_dir, gt['imname']))
                imsize = img.size
                valid_num += 1
                final_label = process_img_bbox(gt['html'], gt['id2bbox'], imsize)
                write_text(final_label, os.path.join(labels_dir, f"{gt['imname'][0:-4]}.md"))
            else:
                print(f"下载失败: {img_url}")
        else:
            print(img_url)
    print(f"处理完成！总共 {valid_num} 张图片")
    

if __name__ == "__main__":
    main()
