"""
This file is used to see if the context of a question is changing when I expand the search range provided to the model.
"""

import json
import glob

# 设定输出文件名
output_filename = "diff_report.txt"

# 使用 with open 确保文件最后被正确关闭
with open(output_filename, "w", encoding="utf-8") as f:

    # 1. 获取当前目录下所有的json文件
    json_files = sorted(glob.glob("512tokens/*.json"))
    if not json_files:
        print("❌ 当前目录下没有找到 .json 文件，请检查路径。", file=f)
        exit()

    print(f"🔍 正在分析 {len(json_files)} 个文件: {json_files}\n", file=f)

    # 2. 读取所有文件数据
    all_data = []
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                all_data.append(json.load(file))
        except Exception as e:
            print(f"⚠️ 读取文件 {file_path} 失败: {e}", file=f)
            exit()

    # 假设所有文件的键结构都一样，取第一个文件的键作为基准
    keys_to_check = all_data[0].keys()
    changed_keys_count = 0

    # 3. 逐个字段对比
    print("-" * 50, file=f)
    for key in keys_to_check:
        # 收集该 key 在所有文件中的值
        values = [data.get(key) for data in all_data]

        # 检查是否所有值都相同 (比较第一个值和剩下的所有值)
        # 注意：这里使用 == 比较列表内容
        first_val = values[0]
        if all(v == first_val for v in values):
            continue  # 如果完全一样，跳过不打印
        else:
            changed_keys_count += 1
            print(f"🔴 发现变化的字段: 【 {key} 】", file=f)

            # 打印每个文件对应的值
            for i, val in enumerate(values):
                # 只显示文件名，不显示完整路径，更清晰
                short_name = json_files[i]
                print(f"   📄 {short_name:<30} -> {val}", file=f)
            print("-" * 50, file=f)

    print(f"\n📊 分析完成！", file=f)
    if changed_keys_count == 0:
        print("✅ 所有文件的内容完全一致，没有发现变化。", file=f)
    else:
        print(f"⚠️ 共发现 {changed_keys_count} 个字段的数据存在差异。", file=f)
