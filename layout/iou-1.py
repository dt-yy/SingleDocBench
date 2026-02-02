import json
from collections import defaultdict
import os
import glob

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1['x'], box1['y'], box1['width'], box1['height']
    x2, y2, w2, h2 = box2['x'], box2['y'], box2['width'], box2['height']
    
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def extract_boxes_from_data(data_dict):
    boxes_by_image = {}
    for image_url, content in data_dict.items():
        if not isinstance(content, dict):
            continue
        boxes_list = content.get("boxes", [])
        if not isinstance(boxes_list, list):
            continue
        
        boxes = []
        for item in boxes_list:
            if not isinstance(item, dict):
                continue
            # 直接使用 x, y, width, height
            x = item.get("x")
            y = item.get("y")
            w = item.get("width")
            h = item.get("height")
            label = item.get("label")
            
            if x is None or y is None or w is None or h is None or not label:
                continue
            
            bbox = {
                "x": float(x),
                "y": float(y),
                "width": float(w),
                "height": float(h)
            }
            boxes.append({
                "bbox": bbox,
                "label": str(label)
            })
        
        if boxes:
            boxes_by_image[image_url] = boxes
    
    return boxes_by_image

def load_boxes_from_jsonl(file_path):
    all_boxes_by_image = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                
                boxes_dict = None
                eval_conv = obj.get("evaluation", {}).get("conversation_evaluation")
                ref_conv = obj.get("reference_evaluation", {}).get("conversation_evaluation")
                if isinstance(eval_conv, dict):
                    boxes_dict = eval_conv
                elif isinstance(ref_conv, dict):
                    boxes_dict = ref_conv
                
                if boxes_dict is not None:
                    boxes_by_image = extract_boxes_from_data(boxes_dict)
                    all_boxes_by_image.update(boxes_by_image)
                else:
                    print(f"Warning: Line {line_num} has no valid conversation_evaluation or reference_evaluation.")
            except Exception as e:
                print(f"Error parsing line {line_num} in {file_path}: {e}")
                continue
    
    return all_boxes_by_image

def calculate_micro_f1(gt_boxes, pred_boxes, iou_threshold=0.75):
    gt_by_label = defaultdict(list)
    pred_by_label = defaultdict(list)
    
    for box in gt_boxes:
        gt_by_label[box["label"]].append(box["bbox"])
    for box in pred_boxes:
        pred_by_label[box["label"]].append(box["bbox"])
    
    tp = fp = fn = 0
    all_labels = set(gt_by_label.keys()) | set(pred_by_label.keys())
    
    for label in all_labels:
        gts = gt_by_label[label]
        preds = pred_by_label[label]
        matched_gt = [False] * len(gts)
        
        for pred in preds:
            best_iou = -1
            best_idx = -1
            for i, gt in enumerate(gts):
                if matched_gt[i]:
                    continue
                iou = compute_iou(gt, pred)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= iou_threshold and best_idx != -1:
                tp += 1
                matched_gt[best_idx] = True
            else:
                fp += 1
        
        fn += sum(1 for m in matched_gt if not m)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

# 新增的测试模块
def count_labels(boxes_by_image):
    label_count = defaultdict(int)
    for boxes in boxes_by_image.values():
        for box in boxes:
            label = box["label"]
            label_count[label] += 1
    for label, count in label_count.items():
        print(f"Label: {label}, Count: {count}")

def main():
    gt_file = '/home/shijiayong/code/1/布局试标gt (1).jsonl'
    pred_dir = '/home/shijiayong/code/1/2/'
    
    print("Loading ground truth...")
    gt_boxes_by_image = load_boxes_from_jsonl(gt_file)
    print(f"Loaded {len(gt_boxes_by_image)} images from GT.")
    
    pred_files = glob.glob(os.path.join(pred_dir, "*.jsonl"))
    if not pred_files:
        print(f"No .jsonl files found in {pred_dir}")
        return

    for pred_file in sorted(pred_files):
        student_name = os.path.splitext(os.path.basename(pred_file))[0]
        print(f"\n{student_name}")  # 打印学生名（即文件名）
        
        pred_boxes_by_image = load_boxes_from_jsonl(pred_file)
        print(f"  Predicted on {len(pred_boxes_by_image)} images.")
        
        # 调用测试模块（可选），仅作用于预测数据
        # 注释掉下面这行即可禁用该模块
        #count_labels(pred_boxes_by_image)
        
        evaluated_count = high_score_count = 0
        for image_url in pred_boxes_by_image:
            gt_list = gt_boxes_by_image.get(image_url, [])
            pred_list = pred_boxes_by_image[image_url]
            f1 = calculate_micro_f1(gt_list, pred_list, iou_threshold=0.75)
            
            print(f"Image: {image_url}, F1={f1:.4f}")
            evaluated_count += 1
            
            if f1 > 0.95:
                high_score_count += 1
        
        if evaluated_count == 0:
            print("该同学未提交任何有效图片，无法评分。")
        else:
            ratio = high_score_count / evaluated_count
            percentage = ratio * 100
            if ratio >= 0.9:
                print(f"该同学的高分图片占比为 {percentage:.2f}%，恭喜，该同学合格了！！")
            else:
                print(f"该同学的高分图片占比为 {percentage:.2f}%，未达到90%的标准。")

if __name__ == "__main__":
    main()