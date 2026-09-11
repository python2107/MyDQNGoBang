# MyDQNGoBang

> 五子棋 DQN 自对弈训练项目（中文说明）

本仓库实现了一个基于 DQN（Deep Q-Network）的五子棋自对弈训练框架，包含三种运行模式：

- GUI：基于 Tkinter 的可视化训练界面（GUI/main.py）。适合观察训练过程、手动保存/加载权重。
- NoGUI/Console：纯命令行训练脚本（NoGUI/Console/main.py），适合长期在服务器/终端运行并自动存档。
- NoGUI/Flask：带简单网页展示的后台训练服务（NoGUI/Flask/main.py），模块加载即开始训练，可通过网页查看训练进度和棋盘。

代码使用 Python（以 3.8+ 为佳），主要依赖：PyTorch、NumPy，Flask（仅用于 NoGUI/Flask）。

---

## 快速开始

1. 克隆仓库：

   git clone https://github.com/python2107/MyDQNGoBang.git
   cd MyDQNGoBang

2. 安装依赖（建议使用虚拟环境）：

- 若只运行命令行或 GUI：

  pip install torch numpy

- 若运行 Flask Web 版：

  pip install -r NoGUI/Flask/requirements.txt

说明：PyTorch 请根据你的平台（CUDA / CPU）从 https://pytorch.org/ 选择合适的安装命令。

---

## 目录结构（简要）

- GUI/main.py              - Tkinter 可视化训练界面
- NoGUI/Console/main.py   - 纯训练命令行脚本
- NoGUI/Flask/main.py     - Flask 后台训练并提供网页展示
- NoGUI/Flask/requirements.txt - Flask 版依赖清单
- LICENSE

---

## 使用说明

### GUI（可视化）

运行：

  python GUI/main.py

功能要点：
- 点击“开始训练”开始自对弈训练；可暂停、继续、停止。
- 支持“后台极速模式”不刷新棋盘以节省显示开销并加速训练。
- 可通过“加载轮数”输入框加载 `json/bogang/<轮数>/black.json` 和 `white.json`（GUI 使用 JSON 存储权重）。
- 点击“保存权重”会将当前轮数的权重保存到 `json/bogang/<轮数>/`。

注意：GUI 版本为了兼容简单的跨平台运行，保存格式为 JSON，体积和速度比不上二进制的 torch 保存方式。

### NoGUI/Console（命令行训练）

运行：

  python NoGUI/Console/main.py [--episodes N] [--log-interval M] [--board-size S] [--seed SEED] [--weight-root PATH]

示例：

  python NoGUI/Console/main.py --episodes 5000 --log-interval 20

功能要点：
- 适合在服务器上长时间训练。
- 自动加载最新存档（若存在），并在达到 AUTO_SAVE_INTERVAL（默认 500 局）时自动保存。
- 权重使用 PyTorch 二进制文件保存：`json/bogang/<轮数>/black.pt` 和 `white.pt`，并包含 `meta.json`。
- 会保留最多 20 份存档（可在脚本中调整 MAX_SAVES）。

### NoGUI/Flask（网页展示）

运行：

  cd NoGUI/Flask
  pip install -r requirements.txt
  python main.py

然后打开浏览器访问：

  http://127.0.0.1:5000/

特点：
- 后台训练线程会在模块加载时自动启动（即便无人访问也会训练）。
- 提供 `/api/state` 接口返回当前训练快照，网页定时拉取并渲染。
- 同样在 `json/bogang/` 下保存二进制权重和 meta 信息。

---

## 权重和存档说明

- 存档根目录：`json/bogang/`（默认，可通过命令行参数覆盖）。
- GUI：每个存档目录下使用 `black.json`、`white.json`（JSON 格式，较大且慢）。
- Console/Flask：每个存档目录下使用 `black.pt`、`white.pt`（PyTorch 二进制，推荐在长期训练中使用），并有 `meta.json` 记录一些元信息。
- 存档命名为完成的局数（例如 `json/bogang/1500/`）。程序会按数字顺序管理并清理最旧的存档以保持不超过 MAX_SAVES。

---

## 可配置项（常见）

- BOARD_SIZE：棋盘大小，默认 13（代码中可更改或通过命令行参数覆盖）。
- AUTO_SAVE_INTERVAL：自动保存的局数间隔（默认 500）。
- MEMORY_CAPACITY、BATCH_SIZE、LR、GAMMA、EPSILON_DECAY 等均在脚本顶部定义为超参数，可按需调优。

---

## 代码说明（高层）

- 游戏环境：GoBangGame 实现了棋盘状态、合法动作、落子规则、胜负判断、棋型奖励与对手威胁检测。
- DQN网络：简单的全连接网络（多层全连接 + ReLU），输入为扁平化棋盘，输出为每个格子的 Q 值。
- 训练策略：自对弈，两方均使用 DQNAgent；经验缓存（ReplayBuffer）做随机采样，使用目标网络（target network）并定期同步。
- 奖励设计：结合九宫格区域奖励（鼓励靠中间/关键区域）、棋型奖励（活四、冲四、活三等）与对手威胁惩罚。

---

## 开发与贡献

欢迎提交 Issue 或 Pull Request：
- 如果你改进了网络结构、奖励函数或训练策略，欢迎提交 PR。
- 若希望加入对局回放、对弈对外接口或强化训练监控（TensorBoard / WandB），也欢迎讨论。

---

## 常见问题（FAQ）

Q: 为什么训练���慢？
- A: 如果使用 CPU，训练会比较慢；建议安装支持 CUDA 的 PyTorch 并在带 GPU 的机器上运行。
- A: GUI 模式默认会频繁刷新界面，开启“后台极速模式”可以显著提升速度。

Q: 权重格式不兼容怎么办？
- A: GUI 使用 JSON 存储，Console/Flask 使用 PyTorch二进制（.pt）；在不同模式间切换加载时需注意格式是否匹配。

---

## 许可

本仓库包含 LICENSE 文件，请查看 LICENSE 了解开源许可和使用条款。

---

感谢使用与关注！如需我把此 README 提交到仓库（创建 README.md），我可以帮你完成提交。
