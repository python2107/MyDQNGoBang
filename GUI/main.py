# ============================================================
#  五子棋 GUI 对弈程序 —— 适配 main.py（控制台训练版）的 JSON 存档
#
#  读取路径：json/gobang/{episode}/qgnn13/{black,white,meta}.json
#
#  与训练脚本严格保持一致的部分：
#    * 棋盘 13 路，输入 13*13=169 维展平
#    * 状态编码：黑棋=+1，白棋=-1（绝对视角，不做翻转）
#    * 动作编码：idx = row * 13 + col
#    * 网络结构：169 -> 128 -> 128 -> 169 (ReLU)
#    * 校验字段：fmt=1 / board=13 / hidden=128
# ============================================================
import os
import json
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 与训练脚本一致的配置 ====================
BOARD_SIZE  = 13
HIDDEN      = 128
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
STATE_SIZE  = ACTION_SIZE
FMT_VERSION = 1

SAVE_ROOT = "json/gobang"
SUB_DIR   = "qgnn13"

# ==================== 界面参数 ====================
CELL     = 42
MARGIN   = 36
BOARD_PX = MARGIN * 2 + CELL * (BOARD_SIZE - 1)


# ------------------------------------------------------------
#  网络结构（必须与训练脚本完全相同，否则权重加载失败）
# ------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SIZE, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, ACTION_SIZE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ------------------------------------------------------------
#  推理 Agent（只做贪心决策，不做任何探索）
# ------------------------------------------------------------
class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.model.eval()
        self.epsilon = 0.0
        self.updates = 0
        self.loaded = False

    def load_json(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到权重文件：\n{path}")

        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        # —— 与训练脚本 save_json / load_json 相同的校验 ——
        if d.get("fmt") != FMT_VERSION:
            raise ValueError(f"{path}\nfmt 不匹配：{d.get('fmt')} != {FMT_VERSION}")
        if d.get("board") != BOARD_SIZE:
            raise ValueError(f"{path}\nboard 不匹配：{d.get('board')} != {BOARD_SIZE}")
        if d.get("hidden") != HIDDEN:
            raise ValueError(f"{path}\nhidden 不匹配：{d.get('hidden')} != {HIDDEN}")

        state = {k: torch.tensor(v) for k, v in d["model"].items()}
        self.model.load_state_dict(state)
        self.model.eval()

        self.epsilon = float(d.get("epsilon", 0.0))
        self.updates = int(d.get("updates", 0))
        self.loaded = True
        return d

    def act(self, state, valid_actions):
        """state: (13,13) float32；valid_actions: List[int]；返回落子索引。"""
        with torch.no_grad():
            st = torch.FloatTensor(np.asarray(state, dtype=np.float32).flatten())
            st = st.unsqueeze(0).to(self.device)
            q = self.model(st).cpu().numpy().flatten()

        valid_set = set(valid_actions)
        for idx in range(ACTION_SIZE):
            if idx not in valid_set:
                q[idx] = -np.inf
        return int(np.argmax(q))


# ------------------------------------------------------------
#  主界面
# ------------------------------------------------------------
class GoBangGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("五子棋 · DQN 对弈（13 路）")
        self.root.resizable(False, False)

        # 对局状态
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current = 1
        self.winner = None
        self.human_color = 1
        self.last_move = None
        self.history = []
        self.busy = False
        self.game_id = 0

        self.agent_black = Agent()
        self.agent_white = Agent()
        self.model_info = "未载入模型"

        self._build_ui()
        self.refresh_snapshots()

    # ---------------- UI 构建 ----------------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=(8, 6))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="快照：").pack(side=tk.LEFT)
        self.snap_var = tk.StringVar()
        self.snap_combo = ttk.Combobox(top, textvariable=self.snap_var,
                                       width=8, state="readonly")
        self.snap_combo.pack(side=tk.LEFT)
        self.snap_combo.bind("<<ComboboxSelected>>", lambda e: self.load_selected())

        ttk.Button(top, text="刷新", width=6,
                   command=self.refresh_snapshots).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="载入", width=6,
                   command=self.load_selected).pack(side=tk.LEFT)

        ttk.Label(top, text="   模式：").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="human_black")
        self.mode_combo = ttk.Combobox(
            top, textvariable=self.mode_var, width=12, state="readonly",
            values=["human_black", "human_white", "ai_vs_ai"])
        self.mode_combo.pack(side=tk.LEFT)
        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self.new_game())

        ttk.Button(top, text="新对局", width=8,
                   command=self.new_game).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="悔棋", width=6,
                   command=self.undo).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self.root, width=BOARD_PX, height=BOARD_PX,
                                bg="#E3C08D", highlightthickness=0)
        self.canvas.pack(padx=8, pady=(0, 6))
        self.canvas.bind("<Button-1>", self.on_click)

        self.status_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_var, anchor="w",
                  padding=(10, 5)).pack(side=tk.BOTTOM, fill=tk.X)

    # ---------------- 快照管理 ----------------
    @staticmethod
    def find_snapshots():
        if not os.path.isdir(SAVE_ROOT):
            return []
        eps = []
        for d in os.listdir(SAVE_ROOT):
            if d.isdigit() and os.path.isdir(os.path.join(SAVE_ROOT, d, SUB_DIR)):
                eps.append(int(d))
        return sorted(eps)

    def refresh_snapshots(self):
        snaps = self.find_snapshots()
        values = [str(e) for e in snaps]
        self.snap_combo["values"] = values
        if values:
            self.snap_combo.current(len(values) - 1)   # 默认选中最新
            self.load_selected()
        else:
            self.snap_combo.set("")
            self.model_info = "未找到快照"
            self.update_status()
            messagebox.showwarning(
                "未找到存档",
                f"在 {os.path.abspath(SAVE_ROOT)} 下没有找到 "
                f"<局数>/{SUB_DIR}/ 目录。\n请先运行训练脚本产生存档。")

    def load_selected(self):
        s = self.snap_var.get()
        if not s:
            return
        ep = int(s)
        snap = os.path.join(SAVE_ROOT, str(ep), SUB_DIR)
        try:
            db = self.agent_black.load_json(os.path.join(snap, "black.json"))
            dw = self.agent_white.load_json(os.path.join(snap, "white.json"))
        except Exception as e:
            messagebox.showerror("载入失败", str(e))
            return

        eb = float(db.get("epsilon", 0.0))
        ew = float(dw.get("epsilon", 0.0))
        ub = int(db.get("updates", 0))
        uw = int(dw.get("updates", 0))
        self.model_info = (f"快照 {ep} | 黑: ε={eb:.3f} upd={ub} | "
                           f"白: ε={ew:.3f} upd={uw}")

        self.new_game()      # 载入后自动开新局

    # ---------------- 对局控制 ----------------
    def new_game(self):
        mode = self.mode_var.get()
        self.human_color = {"human_black": 1,
                            "human_white": 2,
                            "ai_vs_ai": None}[mode]

        self.game_id += 1
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current = 1
        self.winner = None
        self.last_move = None
        self.history = []
        self.busy = False

        self.draw_board()
        self.update_status()
        self.maybe_ai_move()

    def undo(self):
        if self.busy or not self.history:
            return
        while self.history:
            i, j, p = self.history.pop()
            self.board[i][j] = 0
            self.current = p
            self.winner = None
            if self.human_color is None or self.current == self.human_color:
                break
        self.last_move = self.history[-1][:2] if self.history else None
        self.draw_board()
        self.update_status()

    def on_click(self, event):
        if self.busy or self.winner is not None:
            return
        if self.human_color is None or self.current != self.human_color:
            return

        j = int(round((event.x - MARGIN) / CELL))
        i = int(round((event.y - MARGIN) / CELL))
        if not (0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE):
            return
        if self.board[i][j] != 0:
            return
        self.place(i, j)

    def place(self, i, j):
        if self.winner is not None or self.board[i][j] != 0:
            return

        p = self.current
        self.board[i][j] = p
        self.history.append((i, j, p))
        self.last_move = (i, j)
        self.draw_board()

        if self.check_winner(i, j):
            self.winner = p
            self.update_status()
            return

        if not self.get_valid_actions_idx():
            self.winner = 0
            self.update_status()
            return

        self.current = 3 - p
        self.update_status()
        self.maybe_ai_move()

    # ---------------- AI 调度 ----------------
    def maybe_ai_move(self):
        if self.winner is not None or self.busy:
            return
        if self.human_color is not None and self.current == self.human_color:
            return
        self.busy = True
        gid = self.game_id
        self.root.after(60, lambda: self._do_ai_move(gid))

    def _do_ai_move(self, gid):
        if gid != self.game_id or self.winner is not None:
            self.busy = False
            return

        agent = self.agent_black if self.current == 1 else self.agent_white
        if not agent.loaded:
            self.busy = False
            messagebox.showwarning("未载入模型",
                                   "还没有载入 AI 权重，请先选择快照并点击【载入】。")
            return

        valid = self.get_valid_actions_idx()
        if not valid:
            self.busy = False
            return

        idx = agent.act(self.get_state(), valid)
        i, j = divmod(idx, BOARD_SIZE)
        if self.board[i][j] != 0:          # 兜底，正常不会发生
            i, j = divmod(valid[0], BOARD_SIZE)

        self.busy = False
        self.place(i, j)

    # ---------------- 规则 / 状态 ----------------
    def get_state(self):
        """与训练脚本 get_state 完全一致：黑=+1，白=-1，绝对视角。"""
        s = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == 1:
                    s[i][j] = 1.0
                elif self.board[i][j] == 2:
                    s[i][j] = -1.0
        return s

    def get_valid_actions_idx(self):
        return [i * BOARD_SIZE + j
                for i in range(BOARD_SIZE)
                for j in range(BOARD_SIZE)
                if self.board[i][j] == 0]

    def check_winner(self, i, j):
        player = self.board[i][j]
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            cnt = 1
            for step in (1, -1):
                x, y = i + dx * step, j + dy * step
                while (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE
                       and self.board[x][y] == player):
                    cnt += 1
                    x += dx * step
                    y += dy * step
            if cnt >= 5:
                return True
        return False

    # ---------------- 绘制 ----------------
    def draw_board(self):
        self.canvas.delete("all")
        n = BOARD_SIZE
        x0 = y0 = MARGIN
        x1 = y1 = MARGIN + CELL * (n - 1)

        for k in range(n):
            self.canvas.create_line(x0, y0 + k * CELL, x1, y0 + k * CELL, fill="#6B4A22")
            self.canvas.create_line(x0 + k * CELL, y0, x0 + k * CELL, y1, fill="#6B4A22")

        # 星位
        for si in (3, 6, 9):
            for sj in (3, 6, 9):
                cx, cy = MARGIN + sj * CELL, MARGIN + si * CELL
                self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                                        fill="#6B4A22", outline="")

        # 棋子
        r = CELL * 0.42
        for i in range(n):
            for j in range(n):
                p = self.board[i][j]
                if p == 0:
                    continue
                cx, cy = MARGIN + j * CELL, MARGIN + i * CELL
                if p == 1:
                    self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                            fill="#1A1A1A", outline="#000000")
                else:
                    self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                            fill="#FAFAFA", outline="#888888")

        # 最后一手标记
        if self.last_move:
            i, j = self.last_move
            cx, cy = MARGIN + j * CELL, MARGIN + i * CELL
            self.canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5,
                                    outline="#E53935", width=2)

    def update_status(self):
        if self.winner == 1:
            state = "● 黑棋 获胜！"
        elif self.winner == 2:
            state = "○ 白棋 获胜！"
        elif self.winner == 0:
            state = "平局！"
        else:
            who = "黑棋" if self.current == 1 else "白棋"
            if self.human_color is None:
                who += "（AI）"
            elif self.current == self.human_color:
                who += "（你）"
            else:
                who += "（AI）"
            state = f"轮到 {who} 落子"

        self.status_var.set(f"{state}      {self.model_info}")


# ------------------------------------------------------------
def main():
    root = tk.Tk()
    try:
        # 高 DPI 屏幕下更清晰（Windows 可用，其他平台忽略）
        root.tk.call("tk", "scaling", 1.2)
    except Exception:
        pass
    GoBangGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
