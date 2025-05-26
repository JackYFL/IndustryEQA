import os
import json
import glob

def merge_json_files(input_dir):
    # 获取指定目录下的所有json文件
    json_files = glob.glob(os.path.join(input_dir, '**/*.json'), recursive=True)
    
    # 合并所有数据
    all_data = []
    warnings = []
    errors = []

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 检查是否为列表
                if isinstance(data, list):
                    for idx, item in enumerate(data, 1):
                        # 尝试获取任何可用的标识符
                        item_id = item.get('id', item.get('question_id', idx))
                        
                        # 确保每个项目有path键
                        if 'path' in item:
                            all_data.append(item)
                        else:
                            warnings.append({
                                'file': file_path, 
                                'item_id': item_id, 
                                'message': f"缺少'path'键"
                            })
                else:
                    errors.append({
                        'file': file_path, 
                        'message': "数据不是列表格式"
                    })
        except json.JSONDecodeError:
            errors.append({
                'file': file_path, 
                'message': "无法解析JSON内容"
            })
    
    # 只对有path键的项目进行排序
    all_data.sort(key=lambda x: x.get('path', ''))
    
    # 重新分配question_id
    for idx, item in enumerate(all_data, 1):
        item['question_id'] = idx
    
    # 保存合并后的json文件
    output_file = os.path.join(input_dir, 'merged_output.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    # 打印警告和错误信息
    if warnings:
        print("警告信息:")
        for warning in warnings:
            print(f"文件: {warning['file']}")
            print(f"项目ID: {warning['item_id']}")
            print(f"消息: {warning['message']}")
            print("---")
    
    if errors:
        print("\n错误信息:")
        for error in errors:
            print(f"文件: {error['file']}")
            print(f"消息: {error['message']}")
            print("---")
    
    return output_file

# 使用示例
merged_file = merge_json_files('./data/difficult')
print(f"合并完成，文件保存在：{merged_file}")