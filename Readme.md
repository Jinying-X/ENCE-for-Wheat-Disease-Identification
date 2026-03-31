ENCE-for-Wheat-Disease-Identification
官方 PyTorch 实现 | 论文标题：《ENCE: A Novel Deep Learning Classification Model for Wheat Disease Identification》
提出融合通道先验卷积注意力（CPCA）与高效局部注意力（ELA）的改进EfficientNetV2模型，结合渐进式奇异值分解（SVD）压缩与两阶段训练策略，实现小麦叶片病害的高精度、轻量化识别，辅助农业病害快速诊断。

1. 研究背景与模型定位
小麦作为全球核心粮食作物，其叶片病害（如叶锈病、白粉病、斑枯病、叶枯病）易导致光合效率骤降、产量损失达10%-30%，传统人工检测存在耗时久、主观性强、田间覆盖范围有限的问题。现有深度学习方法面临三大核心挑战：

局部病灶特征模糊：早期病害（如白粉病初期白色霉层、叶锈病微小孢子堆）难以与健康组织区分；

全局纹理建模效率低：健康叶片的平行叶脉与病害区域的纹理差异难以高效建模；

相似病害区分难：叶锈病与条锈病、斑枯病与叶枯病形态相似，易混淆。

本文提出ENCE（EfficientNetV2 with Channel Prior and Efficient Local Attention）模型，通过双重注意力机制融合与渐进式压缩技术，解决小麦叶片病害识别的核心问题，最终在自建小麦叶片病害数据集上实现99.69%的分类准确率，为小麦病害田间自动化诊断提供高效、可靠的技术方案。

2. ENCE 核心创新点
2.1 双重注意力机制融合：CPCA + ELA
（1）通道先验卷积注意力（CPCA）

CPCA通过串联的通道与空间注意力模块，对小麦叶片特征进行全局-局部联合优化：

通道注意力：通过全局平均池化压缩空间信息，学习不同病害类型的关键通道权重（如白粉病→白色霉层特征通道权重↑、叶锈病→锈色孢子堆特征通道权重↑）；

空间注意力：通过双池化（平均池化+最大池化）与7×7卷积，聚焦病害区域的空间位置，自动定位病斑在叶片上的分布区域，增强对早期微小病斑的敏感性。

（2）高效局部注意力（ELA）

ELA采用轻量化设计，通过一维卷积沿水平和垂直方向独立建模局部空间依赖：

低计算开销：避免传统2D卷积的高成本，以极低计算量提取叶片纹理细节；

纹理敏感：擅长捕捉健康叶片的平行叶脉结构与病害区域的纹理突变，为区分相似病害（如叶锈病与条锈病）提供关键判别特征。

（3）双重注意力的协同机制

text
小麦叶片图像
    ↓
CPCA（全局-局部联合优化）
    ├── 强化病害相关通道（如锈色通道、霉层通道）
    └── 聚焦病斑空间位置
    ↓
ELA（纹理细节增强）
    ├── 水平方向：捕捉叶脉走向
    └── 垂直方向：捕捉病斑边界
    ↓
融合特征 → 高精度分类
2.2 渐进式奇异值分解（SVD）压缩
在训练稳定阶段，对卷积层权重进行低秩近似，保留前8个奇异值以实现智能参数压缩：

（1）渐进式应用策略：

训练初期（Epoch 0-15）：不应用SVD，让模型充分学习特征；

训练中期（Epoch 16-35）：每5轮应用一次SVD，逐步压缩，稳定收敛；

训练后期（Epoch 36-50）：每3轮应用一次SVD，精细压缩，保持精度。

（2）压缩效果：参数量减少约30%，推理速度提升约25%，精度保持仅下降<0.2%。

2.3 两阶段训练策略
（1）阶段一：冻结训练（Epoch 0-4）

训练对象：仅训练CPCA注意力模块、ELA注意力模块和分类头；

冻结对象：EfficientNetV2骨干网络（保护ImageNet预训练特征不被破坏）；

优势：快速适应小麦病害数据（约5轮即可达到85%+准确率），减少训练时间约40%。

（2）阶段二：全参数微调（Epoch 5-49）

训练对象：所有网络层（骨干网络 + 注意力模块 + 分类头）；

学习率策略：学习率减半（精细化调整）；

优势：端到端优化全部特征提取能力，进一步提升对相似病害的区分能力。

3. 实验数据集
3.1 数据集概况
本研究基于自建小麦叶片病害数据集，图像通过6000万像素工业相机采集，并利用Pix2Pix GAN进行数据增强。数据集采用标准的 train/val/test 划分结构，已随项目上传至仓库的 dataset/ 文件夹。

数据集名称	包含类别	图像总数	图像分辨率
Wheat Disease Dataset	健康、叶枯病、斑枯病、叶锈病、白粉病	14,478	统一 resize 至 384×384
3.2 数据集结构
数据集采用标准的 train/val/test 划分结构，文件夹组织如下：

text
train/                      # 训练集
├── Blight/                 # 叶枯病叶片图像
├── Healthy/                # 健康小麦叶片图像
├── Leaf rust/              # 叶锈病叶片图像
├── Powdery mildew/         # 白粉病叶片图像
└── Septoria/               # 斑枯病叶片图像
val/                        # 验证集
├── Blight/
├── Healthy/
├── Leaf rust/
├── Powdery mildew/
└── Septoria/
test/                       # 测试集
├── Blight/
├── Healthy/
├── Leaf rust/
├── Powdery mildew/
└── Septoria/
3.3 病害类别说明
文件夹名称	中文名称	病害特征描述
Blight	叶枯病	叶片枯萎、变色，严重时大面积干枯，影响植株生理功能
Healthy	健康叶片	叶片鲜绿、形态舒展有光泽，整体生长旺盛，无病害迹象
Leaf rust	叶锈病	产生锈色孢子堆，初期黄色或橙色，后期深褐色，使叶片失去光泽
Powdery mildew	白粉病	叶片表面覆盖白色粉状霉层，阻碍气体交换与光合作用
Septoria	斑枯病	初期为褐色小斑点，随病情发展扩大并连接成片，对叶片损害较大
3.4 病害特征与注意力机制对应关系
病害类型	关键视觉特征	CPCA作用	ELA作用
白粉病（Powdery mildew）	白色粉状霉层	增强白色/灰色通道权重	捕捉霉层边缘纹理
叶锈病（Leaf rust）	锈色孢子堆	增强红褐色通道权重	捕捉孢子堆边界
斑枯病（Septoria）	褐色小斑点	增强暗色区域通道	捕捉斑点分布纹理
叶枯病（Blight）	叶片枯萎变色	增强黄色/棕色通道	捕捉枯死区域边界
健康叶片（Healthy）	鲜绿、叶脉清晰	增强绿色通道权重	捕捉平行叶脉纹理
3.5 数据集划分详情
类别	训练集	验证集	测试集	总计
Healthy	2,030	676	680	3,386
Blight	1,748	582	584	2,914
Septoria	1,590	530	532	2,652
Leaf rust	1,550	516	518	2,584
Powdery mildew	1,764	588	590	2,942
总计	8,682	2,892	2,904	14,478
4. 实验环境配置
4.1 依赖安装
推荐使用Anaconda创建虚拟环境，确保依赖版本匹配：

bash
# 1. 创建并激活虚拟环境
conda create -n ence-wheat python=3.10
conda activate ence-wheat

# 2. 安装PyTorch与TorchVision（需适配CUDA版本，示例为CUDA 12.1；CPU用户可替换为cpu版本）
pip install torch==2.6.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其他依赖库
pip install numpy matplotlib opencv-python pandas pillow tqdm timm tensorboard
4.2 硬件要求
GPU（推荐）：NVIDIA GPU（显存≥8GB，如RTX 3060/4060），训练50轮耗时约2-3小时，显存占用峰值≤6GB；

CPU（可选，仅推理）：支持模型推理测试（单张图像推理耗时约0.5-1秒）。若使用CPU进行完整训练，每轮耗时约20-30分钟，50轮总计约17-25小时，不推荐用于完整训练流程，建议仅用于模型测试或小批量数据验证。

训练效率提示：CPU训练速度远低于GPU，建议使用GPU进行模型训练。若仅需测试已训练好的模型，CPU完全满足推理需求。

5. 实验结果
5.1 消融实验结果
模型配置	准确率	对白粉病敏感度	对叶锈病敏感度	参数量
EfficientNetV2（基线）	96.20%	92.5%	93.8%	16.5M
+ CPCA	98.15%	97.2%	97.5%	16.8M
+ ELA	97.85%	96.8%	97.1%	16.6M
+ CPCA + ELA（ENCE）	99.69%	99.2%	99.5%	17.0M
5.2 SVD压缩效果
压缩策略	参数量	推理时间(ms)	准确率	压缩比
无压缩	24.2M	45.2	99.71%	-
固定SVD	17.8M	34.5	99.48%	26.4%
渐进式SVD	17.0M	33.8	99.69%	29.8%
6. 代码使用说明
6.1 模型训练
运行 train.py 脚本启动训练，支持通过参数调整训练配置，示例命令：

bash
python train.py \
  --data_dir ./train \
  --epochs 50 \
  --batch_size 8 \
  --lr 0.005 \
  --weight_decay 1e-5 \
  --save_dir ./weights \
  --device cuda:0
关键参数说明：
参数名	含义	默认值
--data_dir	训练集根目录路径	./train
--epochs	训练轮数	50
--batch_size	批次大小（根据GPU显存调整）	8
--lr	初始学习率	0.005
--freeze_layers	是否启用两阶段训练	True
--freeze_epochs	第一阶段冻结轮数	5
--use_svd	是否启用SVD压缩	True
--use_amp	混合精度训练	True
--save_dir	训练权重保存目录	./weights
--device	训练设备（cuda:0 或 cpu）	cuda:0
训练输出：
训练过程中，模型会自动保存验证集准确率最高的权重至 --save_dir 目录，文件名为 best_model.pth；

每10轮保存一次中间权重至 ./weights/model_epoch{N}.pth；

训练日志（损失值、准确率）实时打印，并通过TensorBoard记录。

启动TensorBoard监控：
bash
tensorboard --logdir=./logs
6.2 模型测试
使用训练好的权重进行测试集评估，运行 test.py 脚本：

bash
python test.py \
  --test_dir ./test \
  --weight_path ./weights/best_model.pth \
  --device cuda:0
6.3 模型预测
使用训练好的权重进行单张小麦叶片图像预测，运行 predict.py 脚本：

bash
python predict.py \
  --image_path ./examples/wheat_leaf_rust.jpg \
  --weight_path ./weights/best_model.pth \
  --device cuda:0
6.4 类别索引映射
项目使用 class_indices.json 文件维护类别与索引的映射关系，内容如下：

json
{
    "Blight": 0,
    "Healthy": 1,
    "Leaf rust": 2,
    "Powdery mildew": 3,
    "Septoria": 4
}
6.5 预训练权重
预训练权重文件 efficientnet_v2_s-dd5fe13b.pth 放置在项目根目录，训练时会自动加载。基于自建数据集训练完成的最优权重保存在 ./weights/best_model.pth，可直接用于预测或微调。

适用场景：仅针对小麦叶片的五类分类（Blight、Healthy、Leaf rust、Powdery mildew、Septoria）。若需扩展其他小麦病害，建议基于此权重微调（冻结浅层注意力模块，仅训练分类头与深层特征融合层，可减少50%以上训练数据量）。
7. 项目文件结构
text
ENCE-for-Wheat-Disease-Identification/
├── train/                      # 训练集
│   ├── Blight/
│   ├── Healthy/
│   ├── Leaf rust/
│   ├── Powdery mildew/
│   └── Septoria/
├── val/                        # 验证集
│   ├── Blight/
│   ├── Healthy/
│   ├── Leaf rust/
│   ├── Powdery mildew/
│   └── Septoria/
├── test/                       # 测试集
│   ├── Blight/
│   ├── Healthy/
│   ├── Leaf rust/
│   ├── Powdery mildew/
│   └── Septoria/
├── model.py                    # 模型定义（含CPCA、ELA、SVD）
├── train.py                    # 模型训练脚本
├── predict.py                  # 模型预测脚本
├── efficientnet_v2_s-dd5fe13b.pth   # ImageNet预训练权重
├── requirements.txt            # 依赖包配置文件
└── README.md                   # 项目说明文档（本文档）
8. 已知问题与注意事项
数据集路径配置：训练前需修改 --data_dir 参数为实际训练集路径，并确保 train、val、test 三个文件夹在同一级目录下；

类别名称对应：代码中的类别名称必须与文件夹名称完全一致（Blight、Healthy、Leaf rust、Powdery mildew、Septoria），注意大小写和空格；

模型配置选择：在 train.py 中可通过 --model_config 参数选择模型配置：

balanced（平衡版）：CPCA和ELA均衡分布，推荐用于一般场景；

light（轻量版）：减少注意力模块数量，适合资源受限设备；

full（完整版）：最大化注意力模块，适合追求极致精度场景；

显存不足处理：

减小 batch_size（如从8减至4）；

启用 --use_amp 混合精度训练（默认已启用）；

调整图像分辨率；

SVD压缩时机：代码中 model.apply_svd_gradually(epoch, epochs) 在训练循环中每5轮自动调用，无需手动干预；

CUDA版本问题：若安装PyTorch时出现CUDA不兼容，可替换为CPU版本（--device cpu），但训练效率会大幅下降；

田间场景适配：若用于实际田间检测，建议先通过数据增强模块扩充数据集，提升模型对复杂环境的适应能力。

9. 引用与联系方式
9.1 引用方式
论文处于投刊阶段，正式发表后将更新BibTeX引用格式，当前可临时引用：

bibtex
@article{ence_wheat_disease,
  title={ENCE: A Novel Deep Learning Classification Model for Wheat Disease Identification},
  author={Xu, Laixiang and Xu, Jinying and Bijani, Madineh and Wu, Longguo and Cai, Zhaopeng and Zhao, Junmin},
  journal={[期刊名称，待录用后补充]},
  year={2026},	
  note={Manuscript submitted for publication}
}
9.2 联系方式
若遇到代码运行问题或学术交流需求，请联系：

邮箱：xujinying@huuc.edu.cn 
GitHub Issue：直接在本仓库提交Issue，会在1-3个工作日内回复。
