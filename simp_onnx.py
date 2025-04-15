#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import onnx
from onnx import shape_inference

try:
    import onnxsim
except ImportError:
    print("错误: 未找到onnxsim包，请先安装：")
    print("pip install onnx-simplifier")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("警告: 未找到onnxruntime包，某些功能可能受限，建议安装：")
    print("pip install onnxruntime")


def get_input_names(model):
    """获取模型的输入节点名称列表"""
    try:
        input_names = [node.name for node in model.graph.input]
        return input_names
    except Exception as e:
        print(f"警告: 无法获取模型输入名称 - {str(e)}")
        return []


def print_model_io_info(model):
    """打印模型的输入输出信息"""
    print("\n模型输入信息:")
    for input in model.graph.input:
        input_shape = []
        if input.type.tensor_type.shape.dim:
            for dim in input.type.tensor_type.shape.dim:
                if dim.dim_param:
                    input_shape.append(dim.dim_param)
                else:
                    input_shape.append(dim.dim_value)
        print(f"  名称: {input.name}, 形状: {input_shape}")

    print("\n模型输出信息:")
    for output in model.graph.output:
        output_shape = []
        if output.type.tensor_type.shape.dim:
            for dim in output.type.tensor_type.shape.dim:
                if dim.dim_param:
                    output_shape.append(dim.dim_param)
                else:
                    output_shape.append(dim.dim_value)
        print(f"  名称: {output.name}, 形状: {output_shape}")


def simplify_onnx(
    input_path,
    output_path,
    check_model=True,
    check_n=3,
    perform_optimization=True,
    skip_fuse_bn=False,
    input_shapes=None,
    dynamic_input_shape=False,
):
    """
    简化ONNX模型

    参数:
        input_path: 输入模型路径
        output_path: 输出模型路径
        check_model: 是否检查简化后的模型
        check_n: 检查模型时使用的随机输入数量
        perform_optimization: 是否执行优化
        skip_fuse_bn: 是否跳过BatchNormalization融合
        input_shapes: 自定义输入形状字典 {input_name: shape}
        dynamic_input_shape: 是否保留动态输入形状 (已废弃，保留兼容)
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入文件 '{input_path}' 不存在")
        return None, False

    print(f"正在加载模型: {input_path}")
    try:
        model = onnx.load(input_path)
    except Exception as e:
        print(f"错误: 无法加载模型 - {str(e)}")
        return None, False

    # 打印模型输入输出信息
    print_model_io_info(model)

    # 执行形状推理
    print("\n执行形状推理...")
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"形状推理警告（模型仍将简化）: {str(e)}")

    # 如果没有指定输入形状，检查是否有动态形状输入
    if input_shapes is None:
        has_dynamic_input = False
        dynamic_inputs = []

        for input in model.graph.input:
            if input.type.tensor_type.shape.dim:
                for dim in input.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        has_dynamic_input = True
                        dynamic_inputs.append(input.name)
                        break

        if has_dynamic_input:
            print("\n警告: 模型包含动态形状输入，需要手动指定输入形状")
            print("使用 --input-shape 参数指定输入形状，格式如下:")

            for input_name in dynamic_inputs:
                print(f'  --input-shape "{input_name} 1,3,224,224"')

            print("\n或根据您的实际需求指定适当的形状")
            print("错误: 需要手动指定输入形状才能继续")
            return None, False

    # 简化模型
    print("\n正在简化模型...")
    try:
        # 注意：dynamic_input_shape 参数已被onnxsim废弃，但保留参数以保持向后兼容
        if dynamic_input_shape:
            print(
                "警告: --dynamic-input-shape 参数已被废弃，onnxsim现在原生支持动态输入形状"
            )

        model_simp, check_ok = onnxsim.simplify(
            model,
            # check_n=check_n if check_model else 0,
            perform_optimization=perform_optimization,
            skip_fuse_bn=skip_fuse_bn,
            input_shapes=input_shapes,
            test_input_shapes=input_shapes,
        )
    except Exception as e:
        error_msg = str(e)
        print(f"错误: 模型简化失败 - {error_msg}")

        # 提供更详细的错误帮助
        if "has dynamic size" in error_msg:
            input_names = get_input_names(model)
            print("\n您需要为模型的动态输入指定形状，例如:")
            for name in input_names:
                print(f'  --input-shape "{name} 1,3,224,224"')
            print("\n请根据您的实际需求指定适当的形状")

        return None, False

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(f"错误: 无法创建输出目录 - {str(e)}")
            return None, False

    # 保存简化后的模型
    try:
        print(f"\n保存简化后的模型到: {output_path}")
        onnx.save(model_simp, output_path)
    except Exception as e:
        print(f"错误: 无法保存模型 - {str(e)}")
        return None, False

    if check_model:
        if check_ok:
            print("模型简化成功，简化前后的模型输出一致！")
        else:
            print("警告：简化前后的模型可能存在差异，请检查输出模型")

    # 打印模型大小对比
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    simplified_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"原始模型大小: {original_size:.2f} MB")
    print(f"简化后模型大小: {simplified_size:.2f} MB")
    print(f"大小减少: {(original_size - simplified_size) / original_size * 100:.2f}%")

    return model_simp, check_ok


def main():
    parser = argparse.ArgumentParser(description="ONNX模型简化工具")
    parser.add_argument("input", help="输入的ONNX模型路径")
    parser.add_argument("-o", "--output", help="输出的ONNX模型路径")
    parser.add_argument("--no-check", action="store_true", help="跳过结果一致性检查")
    parser.add_argument(
        "--check-n", type=int, default=3, help="一致性检查的随机样本数量"
    )
    parser.add_argument("--no-optimization", action="store_true", help="跳过优化步骤")
    parser.add_argument(
        "--skip-fuse-bn", action="store_true", help="跳过BatchNormalization融合"
    )
    parser.add_argument(
        "--dynamic-input-shape",
        action="store_true",
        help="保留动态输入形状 (已废弃，onnxsim现在原生支持动态输入形状)",
    )
    parser.add_argument(
        "--input-shape",
        nargs="+",
        help='指定输入形状，格式: input_name dim1,dim2,...,dimN，例如：--input-shape "input 1,3,224,224"',
    )

    args = parser.parse_args()

    # 如果未指定输出路径，使用默认路径
    if args.output is None:
        filename, ext = os.path.splitext(args.input)
        args.output = f"{filename}_simplified{ext}"

    # 处理输入形状参数
    input_shapes = None
    if args.input_shape:
        input_shapes = {}
        for shape_str in args.input_shape:
            parts = shape_str.split(" ", 1)  # 只在第一个空格处分割
            if len(parts) < 2:
                print(f"错误: 输入形状格式不正确: {shape_str}")
                print("正确格式: input_name dim1,dim2,...,dimN")
                print('示例: --input-shape "input 1,3,224,224"')
                return

            name = parts[0]
            try:
                shape = [int(dim) for dim in parts[1].split(",")]
                input_shapes[name] = shape
            except ValueError:
                print(f"错误: 无法解析维度: {parts[1]}")
                print("维度必须是用逗号分隔的整数，例如: 1,3,224,224")
                return
    print(input_shapes)
    # 执行模型简化
    model_simp, check_ok = simplify_onnx(
        args.input,
        args.output,
        check_model=not args.no_check,
        check_n=args.check_n,
        perform_optimization=not args.no_optimization,
        skip_fuse_bn=args.skip_fuse_bn,
        input_shapes=input_shapes,
        dynamic_input_shape=args.dynamic_input_shape,
    )

    if model_simp is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
