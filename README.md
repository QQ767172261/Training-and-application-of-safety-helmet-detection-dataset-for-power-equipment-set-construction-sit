# 安全帽检测数据集信息表![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/55f59960d60f4bbdb14364c90a7ac7a6.png)

| 信息分类 | 详细内容 |
| ---- | ---- |
| 数据集名称 | Hard Hats 工地安全帽检测数据集 |
| 任务类型 | 2D目标检测 |
| 图像总数量 | 19745张 |
| 类别总数 | 2类 |
| 目标类别 | Hardhat（佩戴安全帽）、NO-Hardhat（未佩戴安全帽） |
| 标注格式 | YOLO TXT归一化矩形框标注，支持导出COCO JSON、VOC XML格式 |
| 数据集划分 | 分为训练集、验证集、测试集，图像与标注文件分目录存放 |
| 标注类型 | 二维边界框检测标注，无分割、关键点标注 |
| 适用场景 | 建筑工地安全监管、厂区人员安全帽佩戴识别、现场视频智能巡检 |
| 开源协议 | CC BY 4.0 |

## YOLOv26 目标检测配套训练代码
### 1. 数据集目录结构
```
hard_hats/
├── train/
│   ├── images
│   └── labels
├── val/
│   ├── images
│   └── labels
├── test/
│   ├── images
│   └── labels
└── data.yaml
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/851a30c81b4343ea8ba8d4b516240625.png)

### 2. data.yaml 配置文件
```yaml
path: ./hard_hats
train: train/images
val: val/images
test: test/images
nc: 2
names:
  0: Hardhat
  1: NO-Hardhat
```
在这里插入图片描述

### 3. 训练脚本 train_hardhat.py
```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov26s.pt")
    model.train(
        data="./hard_hats/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        lr0=0.01,
        patience=15,
        project="runs/train",
        name="yolov26_hardhat",
        exist_ok=True,
        mosaic=1.0
    )
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b0bdc199db60439db8432e6d52bc8e9b.png)

### 4. 评估与推理脚本
```python
from ultralytics import YOLO
# 加载最优权重
model = YOLO("runs/train/yolov26_hardhat/weights/best.pt")
# 验证集指标评估
metrics = model.val(data="./hard_hats/data.yaml")
# 图像推理保存结果
model("site_test.jpg", save=True)
# 视频推理
# model("work_video.mp4", save=True)
```

