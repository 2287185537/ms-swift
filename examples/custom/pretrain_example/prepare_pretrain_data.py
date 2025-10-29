# Copyright (c) Alibaba, Inc. and its affiliates.
"""
准备预训练数据
这个脚本展示如何准备和处理预训练数据
"""

import json
import argparse
from pathlib import Path
from typing import List


def prepare_text_data(input_files: List[str], output_file: str, max_samples: int = None):
    """
    将文本文件转换为 JSONL 格式的预训练数据
    
    Args:
        input_files: 输入文本文件列表
        output_file: 输出 JSONL 文件路径
        max_samples: 最大样本数（可选）
    """
    print(f"正在处理 {len(input_files)} 个输入文件...")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_lines = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for input_file in input_files:
            print(f"  处理: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    
                    # 写入 JSONL 格式
                    json.dump({'text': line}, f_out, ensure_ascii=False)
                    f_out.write('\n')
                    
                    total_lines += 1
                    if max_samples and total_lines >= max_samples:
                        break
            
            if max_samples and total_lines >= max_samples:
                break
    
    print(f"\n✓ 完成! 共处理 {total_lines} 行文本")
    print(f"✓ 输出文件: {output_file}")
    return total_lines


def prepare_pretrain_from_corpus(corpus_dir: str, output_file: str, extensions: List[str] = ['.txt']):
    """
    从语料库目录准备预训练数据
    
    Args:
        corpus_dir: 语料库目录路径
        output_file: 输出 JSONL 文件路径
        extensions: 文件扩展名列表
    """
    corpus_path = Path(corpus_dir)
    
    # 收集所有文本文件
    input_files = []
    for ext in extensions:
        input_files.extend(corpus_path.glob(f'**/*{ext}'))
    
    input_files = [str(f) for f in input_files]
    
    if not input_files:
        print(f"错误: 在 {corpus_dir} 中没有找到任何文本文件")
        return
    
    print(f"找到 {len(input_files)} 个文本文件")
    prepare_text_data(input_files, output_file)


def create_sample_pretrain_data(output_file: str, num_samples: int = 1000):
    """
    创建示例预训练数据（用于测试）
    
    Args:
        output_file: 输出文件路径
        num_samples: 样本数量
    """
    print(f"正在创建 {num_samples} 个示例样本...")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 一些示例文本模板
    sample_texts = [
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "深度学习是机器学习的一个子集，使用多层神经网络来学习数据的表示。",
        "自然语言处理是人工智能领域的一个重要分支，专注于让计算机理解和生成人类语言。",
        "大型语言模型通过在海量文本数据上进行预训练，学习语言的统计规律和语义知识。",
        "Transformer 架构的提出彻底改变了自然语言处理领域，成为当前最流行的模型架构。",
        "预训练-微调范式已经成为自然语言处理任务的标准方法。",
        "注意力机制允许模型在处理序列时聚焦于最相关的部分。",
        "迁移学习使得我们可以将在大规模数据上学到的知识应用到特定任务上。",
        "Python 是人工智能和机器学习领域最流行的编程语言之一。",
        "开源社区推动了人工智能技术的快速发展和广泛应用。"
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            # 循环使用示例文本
            text = sample_texts[i % len(sample_texts)]
            # 添加一些变化
            text = f"样本 {i+1}: {text}"
            
            json.dump({'text': text}, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✓ 完成! 创建了 {num_samples} 个示例样本")
    print(f"✓ 输出文件: {output_file}")
    print(f"\n注意: 这些是示例数据，实际预训练需要使用大规模真实语料。")


def validate_pretrain_data(data_file: str, num_samples: int = 5):
    """
    验证预训练数据格式
    
    Args:
        data_file: 数据文件路径
        num_samples: 显示的样本数
    """
    print(f"正在验证数据文件: {data_file}")
    
    total_lines = 0
    valid_lines = 0
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            total_lines += 1
            try:
                data = json.loads(line)
                if 'text' in data and data['text'].strip():
                    valid_lines += 1
                    if i < num_samples:
                        print(f"\n样本 {i+1}:")
                        print(f"  文本长度: {len(data['text'])} 字符")
                        print(f"  内容预览: {data['text'][:100]}...")
            except json.JSONDecodeError:
                print(f"  警告: 第 {total_lines} 行不是有效的 JSON")
    
    print(f"\n验证结果:")
    print(f"  总行数: {total_lines}")
    print(f"  有效样本: {valid_lines}")
    print(f"  有效率: {valid_lines/total_lines*100:.1f}%")
    
    if valid_lines == total_lines:
        print(f"✓ 数据格式正确!")
    else:
        print(f"⚠ 有 {total_lines - valid_lines} 行数据格式有问题")


def main():
    parser = argparse.ArgumentParser(description='准备预训练数据')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 从文本文件准备数据
    parser_convert = subparsers.add_parser('convert', help='转换文本文件为 JSONL 格式')
    parser_convert.add_argument('--input', nargs='+', required=True, help='输入文本文件')
    parser_convert.add_argument('--output', required=True, help='输出 JSONL 文件')
    parser_convert.add_argument('--max_samples', type=int, help='最大样本数')
    
    # 从目录准备数据
    parser_corpus = subparsers.add_parser('corpus', help='从语料库目录准备数据')
    parser_corpus.add_argument('--input_dir', required=True, help='语料库目录')
    parser_corpus.add_argument('--output', required=True, help='输出 JSONL 文件')
    parser_corpus.add_argument('--extensions', nargs='+', default=['.txt'], help='文件扩展名')
    
    # 创建示例数据
    parser_sample = subparsers.add_parser('sample', help='创建示例数据（用于测试）')
    parser_sample.add_argument('--output', default='./sample_pretrain_data.jsonl', help='输出文件')
    parser_sample.add_argument('--num_samples', type=int, default=1000, help='样本数量')
    
    # 验证数据
    parser_validate = subparsers.add_parser('validate', help='验证数据格式')
    parser_validate.add_argument('--input', required=True, help='数据文件')
    parser_validate.add_argument('--num_samples', type=int, default=5, help='显示的样本数')
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        prepare_text_data(args.input, args.output, args.max_samples)
    elif args.command == 'corpus':
        prepare_pretrain_from_corpus(args.input_dir, args.output, args.extensions)
    elif args.command == 'sample':
        create_sample_pretrain_data(args.output, args.num_samples)
    elif args.command == 'validate':
        validate_pretrain_data(args.input, args.num_samples)
    else:
        parser.print_help()


if __name__ == '__main__':
    # 使用示例
    print("="*60)
    print("预训练数据准备工具")
    print("="*60)
    print("\n使用示例:")
    print("\n1. 创建示例数据（用于测试）:")
    print("   python prepare_pretrain_data.py sample --output ./data/sample.jsonl --num_samples 1000")
    print("\n2. 转换文本文件:")
    print("   python prepare_pretrain_data.py convert --input file1.txt file2.txt --output ./data/train.jsonl")
    print("\n3. 从目录准备数据:")
    print("   python prepare_pretrain_data.py corpus --input_dir ./corpus --output ./data/train.jsonl")
    print("\n4. 验证数据格式:")
    print("   python prepare_pretrain_data.py validate --input ./data/train.jsonl")
    print("="*60)
    print()
    
    main()
