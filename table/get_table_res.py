import os
import re
import json
import argparse
import requests
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any, Tuple
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


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除空格和特殊字符
    
    Args:
        filename (str): 原始文件名
        
    Returns:
        str: 清理后的文件名
    """
    # 移除空格
    filename = filename.replace(" ", "_")
    
    # 移除其他可能导致问题的字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # 移除首尾的点
    filename = filename.strip('.')
    
    return filename


def extract_filename_without_ext(file_path: str) -> str:
    """
    从文件路径中提取文件名（不带扩展名），并清理文件名
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        str: 清理后的文件名（不带扩展名）
    """
    basename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(basename)[0]
    
    # 清理文件名
    sanitized_name = sanitize_filename(name_without_ext)
    
    # 如果清理后文件名为空，使用默认名称
    if not sanitized_name:
        sanitized_name = "unnamed_file"
    
    return sanitized_name


def download_img(img_url: str, save_dir: str) -> str:
    """
    下载图片到指定目录
    
    Args:
        img_url (str): 图片URL
        save_dir (str): 保存目录
        
    Returns:
        str: 下载的图片文件名，如果下载失败返回None
    """
    # 获取原始图片名
    raw_img_name = os.path.basename(img_url)
    
    # 清理图片名
    img_name = sanitize_filename(raw_img_name)
    
    # 如果清理后文件名为空，使用默认名称
    if not img_name:
        img_name = "image.png"
    
    img_path = os.path.join(save_dir, img_name)
    
    if not os.path.exists(img_path):
        try:
            res = requests.get(img_url, timeout=30)
            res.raise_for_status()  # 检查请求是否成功
            with open(img_path, 'wb') as f:
                f.write(res.content)
            return img_name
        except requests.exceptions.RequestException as e:
            print(f"下载失败: {img_url}, 错误: {e}")
            return None
        except Exception as e:
            print(f"下载失败: {img_url}, 错误: {e}")
            return None
    
    return img_name


def write_text(text: str, file_path: str):
    """
    将文本写入文件
    
    Args:
        text (str): 要写入的文本
        file_path (str): 文件路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)


def extract_img_url(text: str) -> str:
    """
    从文本中提取图片URL
    
    Args:
        text (str): 包含图片URL的文本
        
    Returns:
        str: 提取到的图片URL，如果没有则返回空字符串
    """
    # 尝试匹配 ![alt text](url) 格式的图片URL
    patterns = [
        r'!\[.*?\]\((https:.*?\.png)\)',  # PNG格式
        r'!\[.*?\]\((https:.*?\.jpg)\)',  # JPG格式
        r'!\[.*?\]\((https:.*?\.jpeg)\)', # JPEG格式
        r'!\[.*?\]\((https:.*?\.gif)\)',  # GIF格式
        r'!\[.*?\]\((https:.*?\.bmp)\)',  # BMP格式
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return ""


def process_single_data(data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    处理单个数据项
    
    Args:
        data (Dict[str, Any]): 数据项
        
    Returns:
        Tuple[str, Dict[str, Any]]: (图片URL, GT数据)或(None, None)
    """
    angle_str2int = {
        "rotated_angle_0": 0,
        "rotated_angle_90": 270,
        "rotated_angle_180": 180,
        "rotated_angle_270": 90,
    }
    # 检查问卷是否有效
    if data.get('evaluation', {}).get('questionnaire_evaluation', {}) is None:
        return None, None
    if data.get('evaluation', {}).get('questionnaire_evaluation', {}).get('is_invalid_questionnaire') is None:
        return None, None 
    # 获取预测的表格内容
    table_content = data['evaluation']['conversation_evaluation']
    pred = None
    if table_content is None:
        return None, None
    for item in table_content:
        if "boxes" in table_content[item] and table_content[item]["boxes"]:
            pred = table_content[item]["boxes"][0]["attributes"]['content']
            break
    
    if pred is None:
        return None, None
    
    # 获取旋转角度
    try:
        angle = angle_str2int[data['evaluation']['conversation_evaluation']['rotated_angle']]
    except Exception as e:
        angle = 0
    
    # 提取图片URL
    img_url = extract_img_url(data['prompt'])
    if not img_url:
        return None, None
    
    # 获取原始图片名并清理
    raw_img_name = os.path.basename(img_url)
    img_name = sanitize_filename(raw_img_name)
    
    # 如果清理后文件名为空，使用默认名称
    if not img_name:
        img_name = "image.png"
    
    # 获取边界框信息
    try:
        bbox = data['evaluation']['conversation_evaluation'][img_url]['boxes']
        id2bbox = {}
        for box in bbox:
            xmin, ymin, xmax, ymax = box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']
            id2bbox[box['id']] = [xmin, ymin, xmax, ymax]
    except Exception as e:
        id2bbox = {}
    
    # 构建GT数据
    gt = {
        "imname": img_name,
        "raw_imname": raw_img_name,
        "angle": angle,
        "html": pred,
        "otsl": html_to_otsl(pred),
        "id2bbox": id2bbox
    }
    
    return img_url, gt


def process_img_bbox(text: str, id2bbox: Dict[str, List[int]], imsize: Tuple[int, int]) -> str:
    """
    处理图片边界框，将占位符替换为实际坐标
    
    Args:
        text (str): 包含占位符的文本
        id2bbox (Dict[str, List[int]]): ID到边界框的映射
        imsize (Tuple[int, int]): 图片尺寸 (宽度, 高度)
        
    Returns:
        str: 处理后的文本
    """
    W, H = imsize
    for boxid, bbox in id2bbox.items():
        xmin, ymin, xmax, ymax = bbox
        xmin = str(int(xmin / W * 1000)).zfill(3)
        ymin = str(int(ymin / H * 1000)).zfill(3)
        xmax = str(int(xmax / W * 1000)).zfill(3)
        ymax = str(int(ymax / H * 1000)).zfill(3)
        box_str = f"{xmin} {ymin} {xmax} {ymax}"
        text = text.replace(f'<img src="{boxid}">', '<img src="">')
    
    return text


def process_jsonl_file(jsonl_path: str, output_base_dir: str) -> Tuple[int, int]:
    """
    处理单个JSONL文件
    
    Args:
        jsonl_path (str): JSONL文件路径
        output_base_dir (str): 输出基础目录
        
    Returns:
        Tuple[int, int]: (有效图片数量, 总数据数量)
    """
    # 提取文件名（不带扩展名）作为文件夹名，并清理文件名
    folder_name = extract_filename_without_ext(jsonl_path)
    
    # 创建输出目录结构
    output_folder = os.path.join(output_base_dir, folder_name)
    images_dir = os.path.join(output_folder, "images")
    labels_dir = os.path.join(output_folder, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"处理文件: {os.path.basename(jsonl_path)}")
    print(f"清理后文件夹名: {folder_name}")
    print(f"输出到: {output_folder}")
    
    # 读取JSONL文件
    data_list = read_jsonl_file(jsonl_path)
    
    if not data_list:
        print(f"  警告: 文件为空或读取失败")
        return 0, 0
    
    print(f"  读取到 {len(data_list)} 条数据")
    
    # 处理每条数据
    valid_num = 0
    for i, data in enumerate(tqdm(data_list, desc=f"处理 {folder_name}")):
        img_url, gt = process_single_data(data)
        
        if img_url and gt is not None:
            # 下载图片
            img_name = download_img(img_url, images_dir)
            if img_name:
                # 打开图片获取尺寸
                img_path = os.path.join(images_dir, img_name)
                try:
                    img = Image.open(img_path)
                    imsize = img.size
                    
                    # 处理边界框
                    final_label = process_img_bbox(gt['html'], gt['id2bbox'], imsize)
                    
                    # 构建标签文件名（使用清理后的图片名）
                    label_basename = os.path.splitext(img_name)[0]
                    label_filename = f"{label_basename}.md"
                    
                    # 保存标签
                    write_text(final_label, os.path.join(labels_dir, label_filename))
                    
                    valid_num += 1
                except Exception as e:
                    print(f"  图片处理失败: {img_name}, 错误: {e}")
            else:
                print(f"  下载失败: {img_url}")
        else:
            # 可选：记录无效数据
            pass
    
    print(f"  处理完成！有效图片: {valid_num}/{len(data_list)}")
    print()
    
    return valid_num, len(data_list)


def main():
    """
    主函数：处理所有JSONL文件
    """
    #input_path = r"D:\pdf-bench-v2\SingleDocBench\table_data"
    input_path = r"D:\pdf-bench-v2\SingleDocBench\未跑—表格识别知识产权"
    output_path = r"D:\pdf-bench-v2\SingleDocBench\table_result"
    
    # 确保输出基础目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 扫描所有JSONL文件
    jsonl_files = scan_jsonl_files(input_path, recursive=False)
    
    if not jsonl_files:
        print(f"在 {input_path} 中未找到JSONL文件")
        return
    
    print(f"找到 {len(jsonl_files)} 个JSONL文件")
    print("=" * 50)
    
    # 统计信息
    total_valid = 0
    total_data = 0
    
    # 处理每个JSONL文件
    for jsonl_file in jsonl_files:
        valid_num, data_count = process_jsonl_file(jsonl_file, output_path)
        total_valid += valid_num
        total_data += data_count
    
    print("=" * 50)
    print("所有文件处理完成！")
    print(f"总计: {total_valid}/{total_data} 张有效图片")
    print(f"输出目录结构:")
    print(f"  {output_path}/")
    for jsonl_file in jsonl_files:
        folder_name = extract_filename_without_ext(jsonl_file)
        print(f"    ├── {folder_name}/")
        print(f"    │   ├── images/")
        print(f"    │   └── labels/")
    print()
    
    # 生成汇总报告
    summary_file = os.path.join(output_path, "processing_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("JSONL文件处理汇总报告\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"输入目录: {input_path}\n")
        f.write(f"输出目录: {output_path}\n")
        f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"处理文件总数: {len(jsonl_files)}\n")
        f.write(f"总数据条数: {total_data}\n")
        f.write(f"有效图片数: {total_valid}\n")
        f.write(f"有效率: {total_valid/total_data*100:.2f}%\n\n")
        f.write("各文件详情:\n")
        f.write("-" * 40 + "\n")
        
        # 列出所有输出文件夹
        for jsonl_file in jsonl_files:
            folder_name = extract_filename_without_ext(jsonl_file)
            folder_path = os.path.join(output_path, folder_name)
            images_count = 0
            labels_count = 0
            
            if os.path.exists(folder_path):
                images_dir = os.path.join(folder_path, "images")
                labels_dir = os.path.join(folder_path, "labels")
                
                if os.path.exists(images_dir):
                    images_count = len([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
                
                if os.path.exists(labels_dir):
                    labels_count = len([f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))])
            
            f.write(f"{folder_name} (原文件: {os.path.basename(jsonl_file)}):\n")
            f.write(f"  图片数: {images_count}\n")
            f.write(f"  标签数: {labels_count}\n")
    
    print(f"汇总报告已保存到: {summary_file}")


if __name__ == "__main__":
    import time
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")