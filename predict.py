import torch
from torchvision import transforms
from PIL import Image
import json
import os
import numpy as np
from datetime import datetime
try:
    from my_efficientnetv2 import EfficientNetV2

    print("✅ 模型导入成功")
except ImportError as e:
    print(f"❌ 模型导入失败: {e}")
    exit(1)


def test_model():

    print("🎯 开始测试模型...\n")

    # ==================== 配置 ====================

    class Config:

        model_config = "balanced"  # 根据你训练时使用的配置选择: "balanced", "light", "full"
        num_classes = 5
        use_svd = True
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    cfg = Config()

    device = torch.device(cfg.device)
    print(f"使用设备: {device}")
    print(f"模型配置: {cfg.model_config}")

    # 根据配置选择模型结构 - 与训练代码完全相同
    if cfg.model_config == "light":  # 轻量版
        model_cnf = [
            [1, 3, 1, 1, 32, 16, 0, 0, False, True],
            [2, 3, 2, 4, 16, 32, 0, 0, False, True],
            [2, 5, 2, 4, 32, 48, 0, 0, False, True],
            [3, 3, 2, 4, 48, 96, 0, 0.25, True, False],
            [5, 5, 1, 6, 96, 112, 0, 0.25, True, False],
            [6, 3, 2, 6, 112, 192, 0, 0.25, False, True]
        ]
        print("使用：轻量版配置（速度优先）")

    elif cfg.model_config == "full":  # 完整版
        model_cnf = [
            [1, 3, 1, 1, 32, 16, 0, 0, True, False],
            [2, 3, 2, 4, 16, 32, 0, 0, True, False],
            [2, 5, 2, 4, 32, 48, 0, 0, False, True],
            [3, 3, 2, 4, 48, 96, 0, 0.25, True, True],
            [5, 5, 1, 6, 96, 112, 0, 0.25, True, True],
            [6, 3, 2, 6, 112, 192, 0, 0.25, True, True]
        ]
        print("使用：完整版配置（精度优先）")

    else:  # 平衡版（默认）
        model_cnf = [
            [1, 3, 1, 1, 32, 16, 0, 0, True, False],
            [2, 3, 2, 4, 16, 32, 0, 0, False, True],
            [2, 5, 2, 4, 32, 48, 0, 0, False, True],
            [3, 3, 2, 4, 48, 96, 0, 0.25, True, False],
            [5, 5, 1, 6, 96, 112, 0, 0.25, False, True],
            [6, 3, 2, 6, 112, 192, 0, 0.25, True, False]
        ]
        print("使用：平衡版配置（推荐）")


    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ==================== 加载模型 ====================
    print("\n加载模型...")

    # 加载类别映射
    with open('class_indices.json', 'r', encoding='utf-8') as f:
        class_indices = json.load(f)


    model = EfficientNetV2(
        model_cnf=model_cnf,
        num_classes=len(class_indices),
        use_svd=cfg.use_svd
    )

    # 加载权重
    weights_path = "weights/best_model.pth"
    if not os.path.exists(weights_path):
        print(f"❌ 权重文件不存在: {weights_path}")
        return

    # 加载权重（使用strict=False避免严格匹配错误）
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"✅ 加载权重: {weights_path}\n")

    # ==================== 测试配置 ====================
    # 测试目录
    test_dir = r"D:\dataset\test"

    # 文件夹名到类别索引的映射（根据你的实际文件夹结构）
    class_mapping = {
        'Blight': 0,
        'Healthy': 1,
        'Leaf rust': 2,
        'Powdery mildew': 3,
        'Septoria': 4
    }

    if not os.path.exists(test_dir):
        print(f"❌ 测试目录不存在: {test_dir}")
        return

    # ==================== 开始测试 ====================
    print("=" * 60)
    print("开始测试")
    print("=" * 60)

    # 统计变量
    class_correct = np.zeros(len(class_indices))
    class_total = np.zeros(len(class_indices))
    total_correct = 0
    total_images = 0

    # 遍历每个类别
    for folder_name, true_idx in class_mapping.items():
        folder_path = os.path.join(test_dir, folder_name)

        if not os.path.exists(folder_path):
            print(f"⚠ 文件夹不存在: {folder_path}")
            continue

        # 获取所有图片
        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if not image_files:
            print(f"⚠ {folder_name}: 没有图片")
            continue

        print(f"\n📂 测试 {folder_name}...")
        print(f"   图片数量: {len(image_files)}")

        correct = 0

        # 逐张测试
        for file_name in image_files:
            img_path = os.path.join(folder_path, file_name)

            try:
                # 加载和预处理
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)

                # 预测
                with torch.no_grad():
                    output = model(image_tensor)
                    probs = torch.softmax(output, dim=1)
                    predicted = torch.argmax(probs, dim=1).item()

                # 统计
                if predicted == true_idx:
                    correct += 1
                    total_correct += 1

                total_images += 1
                class_total[true_idx] += 1

            except Exception as e:
                print(f"   ❌ 处理失败 {file_name}: {e}")
                continue

        # 显示当前类别的准确率
        accuracy = correct / len(image_files) if len(image_files) > 0 else 0
        class_correct[true_idx] = correct
        print(f"   ✅ 正确: {correct}/{len(image_files)}")
        print(f"   📊 准确率: {accuracy:.2%}")

    # ==================== 显示总体结果 ====================
    print("\n" + "=" * 60)
    print("测试结果统计")
    print("=" * 60)

    # 总体准确率
    overall_accuracy = total_correct / total_images if total_images > 0 else 0
    print(f"\n📈 总体统计:")
    print(f"   总测试图片: {total_images}")
    print(f"   正确预测: {total_correct}")
    print(f"   错误预测: {total_images - total_correct}")
    print(f"   总体准确率: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")

    # 各类别准确率
    print(f"\n📋 各类别准确率:")
    print("-" * 50)

    for folder_name, true_idx in class_mapping.items():
        if class_total[true_idx] > 0:
            acc = class_correct[true_idx] / class_total[true_idx]
            print(
                f"   {folder_name:<15}: {int(class_correct[true_idx]):3d}/{int(class_total[true_idx]):3d} = {acc:.2%}")

    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)

    # 保存结果到文件
    save_results(overall_accuracy, class_correct, class_total, class_mapping, total_images, total_correct)


def save_results(overall_accuracy, class_correct, class_total, class_mapping, total_images, total_correct):
    """保存结果到文本文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    with open(f'测试结果_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("模型测试结果\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("总体统计:\n")
        f.write(f"  总测试图片: {total_images}\n")
        f.write(f"  正确预测: {total_correct}\n")
        f.write(f"  错误预测: {total_images - total_correct}\n")
        f.write(f"  总体准确率: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)\n\n")

        f.write("各类别准确率:\n")
        f.write("-" * 50 + "\n")
        for folder_name, true_idx in class_mapping.items():
            if class_total[true_idx] > 0:
                acc = class_correct[true_idx] / class_total[true_idx]
                f.write(
                    f"{folder_name:<15}: {int(class_correct[true_idx]):3d}/{int(class_total[true_idx]):3d} = {acc:.2%}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"\n💾 结果已保存到: 测试结果_{timestamp}.txt")


if __name__ == "__main__":
    test_model()