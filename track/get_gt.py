import json
from pathlib import Path
from doc_store import DocClient

store = DocClient("http://docs.bigdata.shlab.tech:31080")
anno_tag = "blockpool__000098"
version = "train__mineru_2_5"
output_dir=r"D:\pdf-bench-v2\SingleDocBench\table_gt"
def get_block_metric():
    """
    导出blocks到指定目录
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取并导出blocks
    for block in store.find_blocks(anno_tag):
        try:
            content = store.try_get_content_by_block_id_and_version(block.id, version)
            if content:
                file_path = Path(output_dir) / f"{block.id}.md"
                with open(file_path, 'w', encoding='utf-8') as f:
                    
                    if isinstance(content, (dict, list)):
                        f.write(json.dumps(content, ensure_ascii=False, indent=2))
                    else:
                        f.write(str(content))
        except Exception:
            pass  # 忽略错误继续执行

if __name__ == "__main__":
    get_block_metric()
