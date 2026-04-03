import os
import json
import shutil
from pathlib import Path

DATA_DIR = Path(r"c:\Users\myc\Desktop\数据处理（hping3）\data_hping3_1240Hz")
OUTPUT_DIR = Path(r"c:\Users\myc\Desktop\数据处理（hping3）\dataset_split")

LABEL_MAP = {
    "normal": 0,
    "tcp": 1,
    "udp": 2,
    "icmp": 3
}

ROUND_SPLIT = {
    "train": ["r1", "r2"],
    "val": ["r3"],
    "test": ["r4"]
}

def parse_filename(filename):
    name = filename.replace(".csv", "")
    parts = name.split("_")
    if len(parts) != 3:
        return None
    time_point = parts[0]
    round_num = parts[1]
    label = parts[2]
    if not time_point.startswith("t") or not round_num.startswith("r"):
        return None
    return {
        "time_point": time_point,
        "round": round_num,
        "label": label,
        "label_id": LABEL_MAP.get(label)
    }

def scan_dataset():
    all_files = []
    for time_folder in sorted(DATA_DIR.iterdir()):
        if not time_folder.is_dir():
            continue
        for csv_file in sorted(time_folder.glob("*.csv")):
            parsed = parse_filename(csv_file.name)
            if parsed:
                all_files.append({
                    "filename": csv_file.name,
                    "filepath": str(csv_file.absolute()),
                    **parsed
                })
    return all_files

def split_dataset(files):
    split_result = {key: [] for key in ["train", "val", "test"]}
    for f in files:
        round_name = f["round"]
        for split_name, rounds in ROUND_SPLIT.items():
            if round_name in rounds:
                split_result[split_name].append(f)
                break
    return split_result

def save_split_files(split_result):
    for split_name, file_list in split_result.items():
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for item in file_list:
            src = Path(item["filepath"])
            dst = split_dir / item["filename"]
            if not dst.exists():
                shutil.copy2(src, dst)

def main():
    print("正在扫描数据集...")
    all_files = scan_dataset()
    print(f"共发现 {len(all_files)} 个CSV文件\n")

    print("文件列表：")
    for f in all_files:
        print(f"  {f['filename']} -> 标签:{f['label']}({f['label_id']}) 轮次:{f['round']}")

    print("\n正在划分数据集...")
    split_result = split_dataset(all_files)

    print("\n划分结果统计：")
    for split_name, file_list in split_result.items():
        print(f"  {split_name}: {len(file_list)} 个文件")

    save_split_files(split_result)
    print(f"\n文件已复制到: {OUTPUT_DIR}")

    metadata = {
        "total_files": len(all_files),
        "split": {name: [f["filename"] for f in files] for name, files in split_result.items()},
        "label_map": LABEL_MAP
    }
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, ensure_ascii=False, indent=2)
    print(f"元数据已保存到: {OUTPUT_DIR / 'metadata.json'}")

if __name__ == "__main__":
    main()