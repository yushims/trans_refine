#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本功能：合并两个TSV文件
- 第一个文件 (query.tsv): uttid, transcription
- 第二个文件 (query_all_steps.tsv): uttid, refinedtext  
- 输出文件: uttid, transcription, refinedtext
"""

import pandas as pd
import sys
import os


def merge_tsv_files(file1_path, file2_path, output_path):
    """
    合并两个TSV文件基于uttid
    
    Args:
        file1_path: query.tsv文件路径 (uttid, transcription)
        file2_path: query_all_steps.tsv文件路径 (uttid, refinedtext)
        output_path: 输出文件路径
    """
    try:
        # 读取第一个文件 (query.tsv)
        print(f"正在读取文件: {file1_path}")
        df1 = pd.read_csv(file1_path, sep='\t', header=None, names=['uttid', 'transcription'])
        print(f"文件1包含 {len(df1)} 行数据")
        
        # 读取第二个文件 (query_all_steps.tsv)
        print(f"正在读取文件: {file2_path}")
        df2 = pd.read_csv(file2_path, sep='\t', header=None, names=['uttid', 'refinedtext'])
        print(f"文件2包含 {len(df2)} 行数据")
        
        # 基于uttid合并两个DataFrame
        print("正在合并数据...")
        merged_df = pd.merge(df1, df2, on='uttid', how='inner')
        print(f"合并后包含 {len(merged_df)} 行数据")
        
        # 重新排列列顺序: uttid, transcription, refinedtext
        merged_df = merged_df[['uttid', 'transcription', 'refinedtext']]
        
        # 保存合并后的数据
        print(f"正在保存到文件: {output_path}")
        merged_df.to_csv(output_path, sep='\t', index=False, header=False)
        
        print("合并完成！")
        print(f"统计信息:")
        print(f"  - 原始transcription数据: {len(df1)} 条")
        print(f"  - 原始refinedtext数据: {len(df2)} 条")
        print(f"  - 成功匹配合并: {len(merged_df)} 条")
        
        # 检查是否有未匹配的数据
        missing_in_df2 = set(df1['uttid']) - set(df2['uttid'])
        missing_in_df1 = set(df2['uttid']) - set(df1['uttid'])
        
        if missing_in_df2:
            print(f"  - 在file2中找不到的uttid数量: {len(missing_in_df2)}")
        if missing_in_df1:
            print(f"  - 在file1中找不到的uttid数量: {len(missing_in_df1)}")
            
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}")
        return False
    except Exception as e:
        print(f"错误: {str(e)}")
        return False
    
    return True


def main():
    """主函数"""
    # 设置文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file1_path = os.path.join(script_dir, "query.tsv")
    file2_path = os.path.join(script_dir, "en_query_no_lexical_disfluency.tsv")
    output_path = os.path.join(script_dir, "en_query_no_lexical_disfluency_merged.tsv")
    
    # 检查输入文件是否存在
    if not os.path.exists(file1_path):
        print(f"错误: 找不到文件 {file1_path}")
        return
    
    if not os.path.exists(file2_path):
        print(f"错误: 找不到文件 {file2_path}")
        return
    
    print("=== TSV文件合并脚本 ===")
    print(f"输入文件1 (transcription): {file1_path}")
    print(f"输入文件2 (refinedtext): {file2_path}")
    print(f"输出文件: {output_path}")
    print()
    
    # 执行合并
    success = merge_tsv_files(file1_path, file2_path, output_path)
    
    if success:
        print(f"\n✅ 合并成功！输出文件: {output_path}")
    else:
        print(f"\n❌ 合并失败！")


if __name__ == "__main__":
    main()