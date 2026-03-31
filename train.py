import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import math
from my_efficientnetv2 import EfficientNetV2
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


# ==================== 配置文件 ====================
class Config:
    """所有配置集中在这里"""
    # 基础参数
    num_classes = 5
    epochs = 50
    batch_size = 8
    lr = 0.005
    lrf = 0.01

    # 路径
    data_path = "D:\\AlexNet\\pythonProject\\deep-learning-for-image-processing-master\\pytorch_classification\\Test11_efficientnetV2\\ENCE\\dataset\\train"
    weights = "D:\AlexNet\pythonProject\deep-learning-for-image-processing-master\pytorch_classification\Test11_efficientnetV2\ENCE\权重"

    # 训练策略
    freeze_layers = True
    freeze_epochs = 5
    use_svd = True
    use_amp = True  # 混合精度训练
    accumulation_steps = 4  # 梯度累积步数

    # 模型配置
    model_config = "balanced"  # 可选: "balanced", "light", "full"

    # 设备
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ================================================

def create_weight_mapping():
    """创建权重映射"""
    mappings = {}

    # Stem部分映射
    mappings.update({
        'stem.conv.weight': 'stem.0.weight',
        'stem.bn.weight': 'stem.1.weight',
        'stem.bn.bias': 'stem.1.bias',
        'stem.bn.running_mean': 'stem.1.running_mean',
        'stem.bn.running_var': 'stem.1.running_var',
    })

    # Head部分映射
    mappings.update({
        'head.conv.weight': 'head.0.conv.weight',
        'head.bn.weight': 'head.0.bn.weight',
        'head.bn.bias': 'head.0.bn.bias',
        'head.fc.weight': 'head.1.weight',
        'head.fc.bias': 'head.1.bias',
    })

    # Blocks映射
    for i in range(12):
        # Expand卷积
        mappings[f'blocks.{i}.expand_conv.conv.weight'] = f'blocks.{i}.expand_conv.0.weight'
        mappings[f'blocks.{i}.expand_conv.bn.weight'] = f'blocks.{i}.expand_conv.1.weight'
        mappings[f'blocks.{i}.expand_conv.bn.bias'] = f'blocks.{i}.expand_conv.1.bias'

        # Project卷积
        mappings[f'blocks.{i}.project_conv.conv.weight'] = f'blocks.{i}.project_conv.0.weight'
        mappings[f'blocks.{i}.project_conv.bn.weight'] = f'blocks.{i}.project_conv.1.weight'
        mappings[f'blocks.{i}.project_conv.bn.bias'] = f'blocks.{i}.project_conv.1.bias'

        # SE注意力
        mappings[f'blocks.{i}.se.conv_reduce.weight'] = f'blocks.{i}.se.fc1.weight'
        mappings[f'blocks.{i}.se.conv_expand.weight'] = f'blocks.{i}.se.fc2.weight'

    return mappings


def load_pretrained_weights(model, weights_path, device):
    """加载预训练权重"""
    print("=== 智能权重加载 ===")

    if not os.path.exists(weights_path):
        print(f"⚠ 权重文件不存在: {weights_path}")
        return model

    try:
        weights_dict = torch.load(weights_path, map_location=device)
        model_state_dict = model.state_dict()
        load_weights_dict = {}
        mappings = create_weight_mapping()

        loaded = 0
        skipped = 0

        for pretrain_key, pretrain_val in weights_dict.items():
            # 跳过分类头
            if 'head' in pretrain_key and 'classifier' in pretrain_key:
                skipped += 1
                continue

            # 尝试直接加载
            if pretrain_key in model_state_dict:
                if model_state_dict[pretrain_key].shape == pretrain_val.shape:
                    load_weights_dict[pretrain_key] = pretrain_val
                    loaded += 1
                    continue

            # 尝试映射加载
            if pretrain_key in mappings:
                mapped_key = mappings[pretrain_key]
                if mapped_key in model_state_dict:
                    if model_state_dict[mapped_key].shape == pretrain_val.shape:
                        load_weights_dict[mapped_key] = pretrain_val
                        loaded += 1
                        print(f"  ↳ 映射: {pretrain_key[:30]}... -> {mapped_key[:30]}...")
                        continue

            skipped += 1

        # 加载权重
        if load_weights_dict:
            model.load_state_dict(load_weights_dict, strict=False)
            print(f"✅ 成功加载 {loaded} 个参数")
            print(f"⏭ 跳过 {skipped} 个参数")
        else:
            print("⚠ 未加载任何参数，从头训练")

    except Exception as e:
        print(f"❌ 权重加载失败: {e}")

    return model


def main():
    cfg = Config()

    # 1. 设备配置
    device = torch.device(cfg.device)
    print(f"=== 训练配置 ===")
    print(f"设备：{device} | 类别数：{cfg.num_classes} | 总epoch：{cfg.epochs}")
    print(f"批次大小：{cfg.batch_size} | 初始LR：{cfg.lr}")
    print(f"梯度累积：{cfg.accumulation_steps} | 混合精度：{cfg.use_amp}")
    print(f"模型配置：{cfg.model_config}")

    # 2. 混合精度
    scaler = torch.cuda.amp.GradScaler() if cfg.use_amp and torch.cuda.is_available() else None

    # 3. 创建文件夹
    os.makedirs("./weights", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    tb_writer = SummaryWriter("./logs")

    # 4. 数据加载
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(cfg.data_path)

    # 数据增强
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = MyDataSet(train_images_path, train_images_label, data_transform["train"])
    val_dataset = MyDataSet(val_images_path, val_images_label, data_transform["val"])

    nw = min(os.cpu_count() // 2, 4)
    print(f"数据：训练集{len(train_dataset)} | 验证集{len(val_dataset)} | workers: {nw}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, collate_fn=train_dataset.collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, collate_fn=val_dataset.collate_fn
    )

    # 5. 模型初始化
    print(f"\n=== 模型初始化 ===")

    # 根据配置选择模型结构
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

    model = EfficientNetV2(
        model_cnf=model_cnf,
        num_classes=cfg.num_classes,
        use_svd=cfg.use_svd
    ).to(device)

    # 6. 加载预训练权重
    if cfg.weights and os.path.exists(cfg.weights):
        model = load_pretrained_weights(model, cfg.weights, device)
    else:
        print("⚠ 从头开始训练")

    # 7. 优化器
    if cfg.freeze_layers:
        # 第一阶段：冻结骨干
        for name, param in model.named_parameters():
            if "cpca" in name or "ela" in name or "head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr, weight_decay=1e-4
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    # 学习率调度
    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.epochs)) / 2) * (1 - cfg.lrf) + cfg.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 8. 训练循环
    print(f"\n=== 开始训练 ===\n")
    best_val_acc = 0.0
    unfrozen = False

    for epoch in range(cfg.epochs):
        # 阶段切换
        if cfg.freeze_layers and not unfrozen and epoch >= cfg.freeze_epochs:
            print(f"\n📌 阶段2：解冻所有层")
            for param in model.parameters():
                param.requires_grad = True

            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr * 0.5, weight_decay=1e-4)
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
            unfrozen = True

        # SVD应用
        if cfg.use_svd and epoch % 5 == 0:
            print(f"Epoch {epoch}: 应用SVD...")
            model.apply_svd_gradually(epoch, cfg.epochs)

        # 训练
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            accumulation_steps=cfg.accumulation_steps,
            use_amp=cfg.use_amp,
            scaler=scaler
        )
        scheduler.step()

        # 验证
        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch
        )

        # 记录
        current_lr = optimizer.param_groups[0]['lr']
        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("val_acc", val_acc, epoch)
        tb_writer.add_scalar("lr", current_lr, epoch)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "./weights/best_model.pth")
            print(f"💾 保存最佳 | epoch{epoch} | acc:{val_acc:.4f}")

        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"./weights/model_epoch{epoch + 1}.pth")

        # 打印信息
        phase = "冻结" if epoch < cfg.freeze_epochs and cfg.freeze_layers else "微调"
        print(f"Epoch [{epoch + 1:3d}/{cfg.epochs}] {phase} | "
              f"train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"val loss: {val_loss:.4f} acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")

    # 训练结束
    print(f"\n{'=' * 50}")
    print(f"训练完成！最佳准确率: {best_val_acc:.4f}")
    print(f"权重保存在: ./weights/")
    print(f"{'=' * 50}")

    torch.save(model.state_dict(), "./weights/final_model.pth")
    tb_writer.close()


if __name__ == '__main__':
    # 直接运行，所有配置都在Config类中
    main()