# MyDQNGoBang

> 五子棋 DQN 自对弈训练项目（中文说明）

本仓库是一个基于 DQN（Deep Q-Network）的五子棋自对弈训练框架，目标是让两个智能体在 13x13 棋盘上持续自我对弈并学习更强的落子策略。项目目前包含三种核心运行方式：

- GUI：Tkinter 图形界面训练，适合本地观察训练过程和手动控制。
- NoGUI/Console：纯命令行训练，适合后台挂机训练、长时间跑任务和自动保存。
- NoGUI/Micropython_for_esp32：适用于 ESP32-S3/MicroPython 设备的低资源版本，支持在单片机上训练并保留 JSON 权重格式。

说明：当前仓库中没有 `NoGUI/Flask` 目录；如果旧文档里出现这个目录，属于历史说明或未同步文档。当前维护的主流程为 GUI / Console / ESP32 三种模式。

---

## 1. 项目概览

这个项目的核心思路是：

- 五子棋状态通过一个平铺的一维特征表示，输入到 DQN 网络；
- 每个落子都对应一个动作值 Q(s, a)；
- 两个智能体黑白双方同时学习，互相博弈提高策略；
- 训练会记录权重快照 (`black.json` / `white.json` / `meta.json`) 以便恢复训练；
- 代码中加入了棋型奖励、区域奖励和威胁惩罚，帮助模型更快学会“围堵/进攻/防守”。

---

## 2. 目录结构

```text
MyDQNGoBang/
├── README.md
├── LICENSE
├── GUI/
│   ├── main.py
│   └── requirements.txt
├── NoGUI/
│   ├── Console/
│   │   └── main.py
│   └── Micropython_for_esp32/
│       ├── main.py
│       ├── esp32_has_ulab_v1.29.bin
│       └── ESP32_GENERIC_S3-SPIRAM_OCT-20260824-v1.29.0 (1).bin
└── json/                  # 训练生成的权重/快照目录（第一次训练时自动创建）
```

说明：

- `GUI/main.py`：可视化训练界面，适合电脑本地跑训练；
- `NoGUI/Console/main.py`：无界面训练脚本，适合服务器/后台训练；
- `NoGUI/Micropython_for_esp32/main.py`：ESP32 版训练脚本，适合在资源受限设备上运行；
- `json/gobang/...`：训练存档目录，由程序自动写入，默认保存格式为 JSON。

---

## 3. 运行环境

建议使用 Python 3.8+ / 3.10+，并创建虚拟环境。

### 3.1 通用依赖

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

pip install torch numpy
```

### 3.2 GUI 版额外说明

Tkinter 是 Python 自带的 GUI 库，通常随 Python 安装包一起提供。若在某些 Linux 环境中缺失，可安装：

```bash
sudo apt-get install python3-tk
```

### 3.3 ESP32/MicroPython 版说明

该目录下提供了 MicroPython 固件文件：

- `esp32_has_ulab_v1.29.bin`
- `ESP32_GENERIC_S3-SPIRAM_OCT-20260824-v1.29.0 (1).bin`

这些固件可用于 ESP32-S3 等目标板，前者带 `ulab`，更适合数值计算加速。给 ESP32 烧录 MicroPython 后，再把 `NoGUI/Micropython_for_esp32/main.py` 上传到板子中运行即可。

---

## 4. 快速开始

### 4.1 方式一：GUI 训练（最直观）

```bash
git clone https://github.com/python2107/MyDQNGoBang.git
cd MyDQNGoBang
pip install torch numpy
python GUI/main.py
```

运行后：

- 左侧/上方通常会显示棋盘和训练状态；
- 可点击“开始训练”开始自对弈；
- 可暂停/继续/停止；
- 可手动保存和加载模型；
- 程序会把训练快照保存到 `json/gobang/<局数>/qgnn13/`。

GUI 版保存格式是 JSON，便于可视化和调试；对应文件：

```text
json/gobang/1500/qgnn13/
├── black.json
├── white.json
├── meta.json
```

说明：

- GUI 版为了方便跨平台和展示，采用 JSON 存档；
- 这类文件比 `.pt` 更容易观察和调试；
- 训练中断后，可直接从最新快照恢复继续训练。

---

### 4.2 方式二：命令行训练（服务器/后台最佳）

在项目根目录运行：

```bash
python NoGUI/Console/main.py
```

这会启动无限训练，直到手动 Ctrl+C 停止。默认会自动保存快照。

#### 4.2.1 常见参数

```bash
python NoGUI/Console/main.py --episodes 5000
python NoGUI/Console/main.py --log-interval 20
python NoGUI/Console/main.py --board-size 13 --hidden 256 --seed 42
python NoGUI/Console/main.py --weight-root json/gobang --save-subdir qgnn13
```

参数说明：

- `--episodes`：总训练局数，0 表示无限训练；
- `--log-interval`：每多少局打印一次日志；
- `--board-size`：棋盘大小，默认 13；
- `--hidden`：隐藏层宽度，默认 256；
- `--seed`：随机种子，方便复现实验；
- `--weight-root`：存档根目录；
- `--save-subdir`：每个快照目录下的子目录名，默认 `qgnn13`。

#### 4.2.2 训练日志示例

```text
[第      20 局] 黑胜     8 (40.0%) / 白胜     9 (45.0%) / 平     3 | ε 0.9571 | loss 0.1234 | 经验池   128 | 0.421s/局
```

含义：

- 当前第 20 局；
- 黑胜 8、白胜 9、平 3；
- ε 表示探索率；
- loss 为训练损失；
- 经验池大小表示 replay buffer 中的样本数量。

#### 4.2.3 自动保存规则

脚本默认会在每 500 局自动保存一次（`AUTO_SAVE_INTERVAL = 500`）。

生成目录类似：

```text
json/gobang/500/qgnn13/
├── black.json
├── white.json
├── meta.json
```

最多保留 `MAX_SAVES` 份快照（默认 20），旧快照会被自动删除，防止磁盘被占满。

---

### 4.3 方式三：ESP32 / MicroPython 训练

这是一个适配到单片机的轻量化版本，特点是：

- 通过 `ulab`（当固件中带有 ulab 时）加速矩阵运算；
- 适合在 ESP32-S3 等资源受限平台上训练；
- 存档仍然使用 JSON 格式，和 PC 版兼容；
- 能在设备上自动恢复进度、周期性保存并持续训练。

#### 4.3.1 烧录固件

使用你最合适的工具把 MicroPython 固件烧录进 ESP32。示例：

```bash
esptool.py --chip esp32s3 --port /dev/ttyUSB0 write_flash -z 0x0 esp32_has_ulab_v1.29.bin
```

#### 4.3.2 上传代码和权重

把 `NoGUI/Micropython_for_esp32/main.py` 传到 ESP32 中，并确保其能访问 `SAVE_DIR = "/qgnn13"`。常见目录结构：

```text
/qgnn13/
├── black.json
├── white.json
├── meta.json
```

在 Thonny / mpremote / WebREPL 中运行：

```python
import main
```

启动后，程序会：

- 自动读取 `meta.json` 恢复最新进度；
- 如果没有现成模型，则从随机初始化开始；
- 每隔 `AUTO_SAVE_INTERVAL` 局自动保存；
- 每隔 `LOG_INTERVAL` 局输出训练状态；
- 可通过 Ctrl+C 或重启中断并保留当前进度。

#### 4.3.3 ESP32 版 注意事项

- ESP32 的内存远小于电脑，因此该版使用更小的 batch 和 replay 缓冲；
- `BATCH_SIZE = 4`，`MEMORY_CAPACITY = 300`，这是为了避免内存溢出；
- 该版适合在开发板上演示训练，而非高性能研究训练；
- 如果使用强力版固件（如带 `ulab`），训练会比纯基础固件更快。

---

## 5. 存档与恢复机制

这个项目的权重按“快照”保存。每个快照目录中都包含：

```text
<weight_root>/<episode>/<save_subdir>/
├── black.json
├── white.json
├── meta.json
```

其中：

- `black.json`：黑方 DQN 网络参数；
- `white.json`：白方 DQN 网络参数；
- `meta.json`：训练统计���息，比如 `episode`、`b_wins`、`w_wins`、`draws`。

### 恢复训练

程序会扫描 `json/gobang` 下的所有数字目录，选择最新的有效快照。加载成功后：

- 程序从该快照恢复网络参数；
- 恢复历史胜负记录；
- 继续后续训练，不会从头开始。

例如：

```text
json/gobang/
├── 100/
│   └── qgnn13/
│       ├── black.json
│       ├── white.json
│       └── meta.json
├── 500/
│   └── qgnn13/
│       ├── black.json
│       ├── white.json
│       └── meta.json
└── 1000/
    └── qgnn13/
        ├── black.json
        ├── white.json
        └── meta.json
```

若现在运行训练，程序通常会从 `1000` 目录继续。

---

## 6. 训练原理概述

### 6.1 状态表示

棋盘用 13x13 的二维矩阵表示，然后展平成一维输入：

- 自己落子位置记为 `1`；
- 对手落子位置记为 `-1`；
- 空位记为 `0`。

因此每个状态的输入维度约为：

```text
13 * 13 = 169
```

### 6.2 网络结构

代码中采用的是一个简单的前馈 DQN：

```text
输入 169 -> 隐藏层 256 -> 隐藏层 256 -> 输出 169
```

输出 169 对应棋盘中每一个落点的 Q 值，网络最终选择一个最优落子位置。

### 6.3 奖励设计

奖励并不只是“赢了加分/输了扣分”，而是融合了：

- 九宫格区域奖励：鼓励棋子靠近中心和关键区域；
- 棋型奖励：如活四、冲四、活三、眠三、活二；
- 对手威胁惩罚：对对手即将成型的危险局势做惩罚；
- 胜负判定：一局结束时给出额外的终局奖励。

这让模型更容易学会“进攻”和“制止对手威胁”。

---

## 7. 常见问题（FAQ）

### Q1：训练很慢怎么办？

A：

- CPU 上训练会明显慢；建议使用 NVIDIA GPU 环境并安装对应版本的 PyTorch；
- GUI 模式会刷新界面，界面占用较大，训练速度可能偏慢；
- 目前 `NoGUI/Console` 更适合长时间后台训练；
- 对于 ESP32，应该理解它是“轻量实验/演示训练”，不是高性能实战训练环境。

### Q2：为什么存档目录里是 `black.json`、`white.json`、`meta.json`？

A：因为本项目是双智能体自对弈，每个智能体有一份网络参数；再加上一份训练状态说明文件。这样可以在训练中断后恢复精确状态。

### Q3：GUI 版和 Console 版能互相加载吗？

A：可以在同一套 JSON 规范下大致兼容，但最好保持相同的：

- `BOARD_SIZE`
- `HIDDEN`
- `FMT_VERSION`
- `SAVE_ROOT` / `SAVE_SUBDIR`

如果参数不一致，程序会拒绝加载，避免错误的权重覆盖。

### Q4：我想从指定轮次恢复训练？

A：直接把目标快照目录放回默认保存目录即可，程序会自动加载最新快照；如果你希望手动固定某一轮次，最简单办法是：

- 复制该快照文件到对应目录；
- 或者手动运行脚本并确认 `json/gobang` 下最近目录的内容。

### Q5：我能更改棋盘大小吗？

A：可以。多数脚本都允许通过命令行参数覆盖：

```bash
python NoGUI/Console/main.py --board-size 9
```

不过注意：如果修改棋盘大小，训练结果、网络尺寸和存档兼容性都要重新评估。

---

## 8. 说明与建议

### 8.1 适合场景

- 学习 DQN 与强化学习：适合
- 看训练曲线和交互式调参：GUI 最方便
- 长时间后台训练：Console 最合适
- 低资源硬件部署：ESP32/MicroPython 版本最合适

### 8.2 建议工作流

1. 先用 GUI 或 Console 在本机跑几百轮观察效果；
2. 确认训练稳定后，再用 Console 进行长时间训练；
3. 若需要在嵌入式平台演示或验证，可使用 ESP32 版；
4. 通过 `meta.json` 记录胜率/平局/总轮数，便于比较不同超参数的训练效果。

---

## 9. 许可证

本仓库包含 `LICENSE` 文件，使用前请阅读并遵守其许可条款。

---

## 10. 结语

MyDQNGoBang 的核心价值在于：它把强化学习、五子棋规则、模型保存和视觉化训练都整合在一个比较完整的 Python 项目中，适合：

- 学习 DQN 自对弈；
- 研究棋类强化学习；
- 做训练可视化演示；
- 作为嵌入式 AI 实验平台。

如果你愿意，我还可以继续帮你做两件事中的任意一件：

1. 把 README 再优化成更正式的 GitHub 风格排版；
2. 直接补一份 `运行示意图 + 训练流程图` 的增强版说明文档。
