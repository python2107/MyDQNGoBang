# -*- coding: utf-8 -*-
# ============================================================
#  五子棋 GUI 对弈程序 v3-wx —— 侧边栏布局（宽扁窗口）
#  · 控件全部在棋盘右侧
#  · 热力图：色相 + 颜色深度渐变
#  · 威胁高亮：粗圆环 + 白色描边
#  · wx.GraphicsContext 抗锯齿
#
#  ⚠ HIDDEN = 128 必须与 main.py 一致，否则权重加载失败
# ============================================================
import os
import json
import wx

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 与训练脚本一致的配置（勿改） ====================
BOARD_SIZE  = 13
HIDDEN      = 128           # <<< 必须与 main.py 一致！！！！！
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
STATE_SIZE  = ACTION_SIZE
FMT_VERSION = 1

SAVE_ROOT = "json/gobang"
SUB_DIR   = "qgnn13"

# ==================== 界面参数 ====================
CELL         = 42
MARGIN       = 36
BOARD_PX     = MARGIN * 2 + CELL * (BOARD_SIZE - 1)
SIDEBAR_W    = 190               # 侧栏宽度
BOARD_BG_RGB = (227, 192, 141)
BOARD_BG_HEX = "#E3C08D"
GRID_HEX     = "#6B4A22"

# ==================== 威胁检测颜色 ====================
THREAT_COLORS = {
    'five':        '#FFD700',
    'live_four':   '#FF0000',
    'rush_four':   '#FF6D00',
    'live_three':  '#AAFF00',
    'sleep_three': '#00E5FF',
}
THREAT_PRIORITY = {
    'five': 5, 'live_four': 4, 'rush_four': 3,
    'live_three': 2, 'sleep_three': 1,
}

# ==================== 热力图色相控制点 ====================
HEAT_STOPS = [
    (0.00, (60,  90, 255)),
    (0.25, ( 0, 200, 220)),
    (0.50, (90, 220,  60)),
    (0.75, (255, 200,  0)),
    (1.00, (255,  30,  30)),
]


def heat_fill_rgb(v, alpha_lo=0.20, alpha_hi=0.85):
    v = max(0.0, min(1.0, v))
    rgb = HEAT_STOPS[-1][1]
    for k in range(len(HEAT_STOPS) - 1):
        v0, c0 = HEAT_STOPS[k]
        v1, c1 = HEAT_STOPS[k + 1]
        if v <= v1:
            t = (v - v0) / (v1 - v0) if v1 > v0 else 0.0
            rgb = tuple(c0[i] + (c1[i] - c0[i]) * t for i in range(3))
            break
    alpha = alpha_lo + (alpha_hi - alpha_lo) * v
    return tuple(int(rgb[i] * alpha + BOARD_BG_RGB[i] * (1 - alpha))
                 for i in range(3))


# ------------------------------------------------------------
#  网络结构
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
#  推理 Agent
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

        if d.get("fmt") != FMT_VERSION:
            raise ValueError(f"{path}\nfmt 不匹配：{d.get('fmt')} != {FMT_VERSION}")
        if d.get("board") != BOARD_SIZE:
            raise ValueError(f"{path}\nboard 不匹配：{d.get('board')} != {BOARD_SIZE}")
        if d.get("hidden") != HIDDEN:
            raise ValueError(
                f"{path}\nhidden 不匹配：{d.get('hidden')} != {HIDDEN}\n"
                f"（本 GUI 已锁定 HIDDEN={HIDDEN}，请勿改动）")

        state = {k: torch.tensor(v) for k, v in d["model"].items()}
        self.model.load_state_dict(state)
        self.model.eval()

        self.epsilon = float(d.get("epsilon", 0.0))
        self.updates = int(d.get("updates", 0))
        self.loaded = True
        return d

    def qvalues(self, state, valid):
        with torch.no_grad():
            st = torch.FloatTensor(
                np.asarray(state, dtype=np.float32).flatten()
            ).unsqueeze(0).to(self.device)
            q = self.model(st).cpu().numpy().flatten()
        valid_set = set(valid)
        for i in range(ACTION_SIZE):
            if i not in valid_set:
                q[i] = -np.inf
        return q

    def act(self, state, valid):
        return int(np.argmax(self.qvalues(state, valid)))


# ------------------------------------------------------------
#  棋盘 Panel
# ------------------------------------------------------------
class BoardPanel(wx.Panel):
    def __init__(self, parent, gui):
        super().__init__(parent, size=(BOARD_PX, BOARD_PX))
        self.gui = gui
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetMinSize((BOARD_PX, BOARD_PX))
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_click)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda e: None)

    def on_paint(self, evt):
        dc = wx.AutoBufferedPaintDC(self)
        dc.SetBackground(wx.Brush(wx.Colour(*BOARD_BG_RGB)))
        dc.Clear()
        gc = wx.GraphicsContext.Create(dc)
        if gc is None:
            return
        gc.SetAntialiasMode(wx.ANTIALIAS_DEFAULT)
        self.gui.draw_board(gc)

    def on_click(self, evt):
        self.gui.handle_click(evt.GetX(), evt.GetY())


# ------------------------------------------------------------
#  主窗口
# ------------------------------------------------------------
class GoBangFrame(wx.Frame):
    def __init__(self):
        super().__init__(
            None,
            title=f"五子棋 · DQN 对弈 v3-wx（HIDDEN={HIDDEN}）",
            style=wx.DEFAULT_FRAME_STYLE & ~wx.MAXIMIZE_BOX)

        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current = 1
        self.winner = None
        self.human_color = 1
        self.last_move = None
        self.history = []
        self.busy = False
        self.game_id = 0
        self._ai_prediction = None

        self.agent_black = Agent()
        self.agent_white = Agent()
        self.model_info = "未载入模型"

        self._build_ui()
        self._refresh_snapshots()
        self.Centre()

    # ---------------- UI ----------------
    def _build_ui(self):
        panel = wx.Panel(self)
        panel.SetBackgroundColour(wx.Colour(240, 240, 240))
        outer = wx.BoxSizer(wx.VERTICAL)

        main_row = wx.BoxSizer(wx.HORIZONTAL)

        # —— 棋盘（左） ——
        self.board_panel = BoardPanel(panel, self)
        main_row.Add(self.board_panel, 0, wx.ALL, 6)

        # —— 侧边栏（右） ——
        side = wx.BoxSizer(wx.VERTICAL)

        side.Add(wx.StaticText(panel, label="快照"), 0, wx.BOTTOM, 2)
        self.snap_combo = wx.ComboBox(panel, style=wx.CB_READONLY,
                                      size=(SIDEBAR_W, -1))
        self.snap_combo.Bind(wx.EVT_COMBOBOX, self.on_snapshot_change)
        side.Add(self.snap_combo, 0, wx.BOTTOM, 6)

        br1 = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_refresh = wx.Button(panel, label="刷新",
                                     size=((SIDEBAR_W - 6) // 2, -1))
        self.btn_refresh.Bind(wx.EVT_BUTTON, lambda e: self._refresh_snapshots())
        br1.Add(self.btn_refresh, 0)
        self.btn_load = wx.Button(panel, label="载入",
                                  size=((SIDEBAR_W - 6) // 2, -1))
        self.btn_load.Bind(wx.EVT_BUTTON, self.on_load)
        br1.Add(self.btn_load, 0, wx.LEFT, 6)
        side.Add(br1, 0, wx.BOTTOM, 12)

        side.Add(wx.StaticText(panel, label="模式"), 0, wx.BOTTOM, 2)
        self.mode_combo = wx.ComboBox(
            panel, value="human_black",
            choices=["human_black", "human_white", "ai_vs_ai"],
            style=wx.CB_READONLY, size=(SIDEBAR_W, -1))
        self.mode_combo.Bind(wx.EVT_COMBOBOX, self.on_mode_change)
        side.Add(self.mode_combo, 0, wx.BOTTOM, 6)

        br2 = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_new = wx.Button(panel, label="新对局",
                                 size=((SIDEBAR_W - 6) // 2, -1))
        self.btn_new.Bind(wx.EVT_BUTTON, lambda e: self.new_game())
        br2.Add(self.btn_new, 0)
        self.btn_undo = wx.Button(panel, label="悔棋",
                                  size=((SIDEBAR_W - 6) // 2, -1))
        self.btn_undo.Bind(wx.EVT_BUTTON, lambda e: self.undo())
        br2.Add(self.btn_undo, 0, wx.LEFT, 6)
        side.Add(br2, 0, wx.BOTTOM, 12)

        self.cb_heat = wx.CheckBox(panel, label="AI 思考热力图")
        self.cb_heat.SetValue(True)
        self.cb_heat.Bind(wx.EVT_CHECKBOX, lambda e: self.board_panel.Refresh())
        side.Add(self.cb_heat, 0, wx.BOTTOM, 4)

        self.cb_threat = wx.CheckBox(panel, label="威胁高亮")
        self.cb_threat.SetValue(True)
        self.cb_threat.Bind(wx.EVT_CHECKBOX, lambda e: self.board_panel.Refresh())
        side.Add(self.cb_threat, 0, wx.BOTTOM, 12)

        side.Add(wx.StaticText(panel, label="图例"), 0, wx.BOTTOM, 4)
        for name, key in [("五连", 'five'), ("活四", 'live_four'),
                          ("冲四", 'rush_four'), ("活三", 'live_three'),
                          ("眠三", 'sleep_three')]:
            lr = wx.BoxSizer(wx.HORIZONTAL)
            lr.Add(wx.StaticBitmap(
                panel, bitmap=self._legend_bitmap(THREAT_COLORS[key])),
                0, wx.ALIGN_CENTER_VERTICAL)
            lr.Add(wx.StaticText(panel, label="  " + name),
                   0, wx.ALIGN_CENTER_VERTICAL)
            side.Add(lr, 0, wx.BOTTOM, 3)

        side.AddStretchSpacer()

        self.model_info_label = wx.StaticText(panel, label="")
        f = self.model_info_label.GetFont()
        f.SetPointSize(max(8, f.GetPointSize() - 1))
        self.model_info_label.SetFont(f)
        self.model_info_label.SetForegroundColour(wx.Colour(80, 80, 80))
        side.Add(self.model_info_label, 0, wx.EXPAND | wx.TOP, 6)

        main_row.Add(side, 0, wx.TOP | wx.BOTTOM | wx.RIGHT, 6)

        outer.Add(main_row, 0, wx.EXPAND)

        # —— 状态栏（底部通栏） ——
        self.status_text = wx.StaticText(panel, label="")
        f = self.status_text.GetFont()
        f.SetPointSize(f.GetPointSize() + 1)
        f.MakeBold()
        self.status_text.SetFont(f)
        outer.Add(self.status_text, 0, wx.EXPAND | wx.ALL, 8)

        panel.SetSizer(outer)

        # —— 显式计算窗口尺寸（不依赖 Fit 的自动推算） ——
        # 宽度 = 棋盘区 + 侧栏区 + 各种边距
        #   棋盘：BOARD_PX + 左右各 6
        #   侧栏：SIDEBAR_W + 右 6
        #   再加一个 40 冗余（边框 + 滚动条余量）
        win_w = BOARD_PX + SIDEBAR_W + 6 * 3 + 40
        # 高度 = 棋盘区 + 底部状态栏 + 上下边距
        win_h = BOARD_PX + 6 * 2 + 46
        self.SetClientSize((win_w, win_h))
        # 锁定初始 / 最小尺寸，防止用户拖动缩到看不见
        self.SetSizeHints(win_w, win_h, -1, -1)

    def _legend_bitmap(self, color_hex):
        bmp = wx.Bitmap(18, 18)
        mdc = wx.MemoryDC(bmp)
        mdc.SetBackground(wx.Brush(wx.Colour(*BOARD_BG_RGB)))
        mdc.Clear()
        gc = wx.GraphicsContext.Create(mdc)
        if gc:
            gc.SetAntialiasMode(wx.ANTIALIAS_DEFAULT)
            gc.SetBrush(wx.TRANSPARENT_BRUSH)
            gc.SetPen(wx.Pen(wx.Colour(color_hex), 3))
            gc.DrawEllipse(3, 3, 12, 12)
        mdc.SelectObject(wx.NullBitmap)
        return bmp

    # ---------------- 事件 ----------------
    def on_snapshot_change(self, evt):
        pass

    def on_mode_change(self, evt):
        self.new_game()

    def on_load(self, evt):
        s = self.snap_combo.GetValue()
        if not s:
            return
        try:
            ep = int(s)
        except ValueError:
            return
        snap = os.path.join(SAVE_ROOT, str(ep), SUB_DIR)
        try:
            db = self.agent_black.load_json(os.path.join(snap, "black.json"))
            dw = self.agent_white.load_json(os.path.join(snap, "white.json"))
        except Exception as e:
            wx.MessageBox(str(e), "载入失败", wx.OK | wx.ICON_ERROR)
            return

        eb = float(db.get("epsilon", 0.0))
        ew = float(dw.get("epsilon", 0.0))
        ub = int(db.get("updates", 0))
        uw = int(dw.get("updates", 0))
        self.model_info = (f"快照 {ep}  (H={HIDDEN})\n"
                           f"黑  ε={eb:.3f}  upd={ub}\n"
                           f"白  ε={ew:.3f}  upd={uw}")
        self.model_info_label.SetLabel(self.model_info)
        self.Layout()
        self.new_game()

    # ---------------- 快照 ----------------
    @staticmethod
    def find_snapshots():
        if not os.path.isdir(SAVE_ROOT):
            return []
        eps = []
        for d in os.listdir(SAVE_ROOT):
            if d.isdigit() and os.path.isdir(os.path.join(SAVE_ROOT, d, SUB_DIR)):
                eps.append(int(d))
        return sorted(eps)

    def _refresh_snapshots(self):
        values = [str(e) for e in self.find_snapshots()]
        self.snap_combo.SetItems(values)
        if values:
            self.snap_combo.SetSelection(len(values) - 1)
            self.on_load(None)
        else:
            self.snap_combo.SetValue("")
            self.model_info = "未找到快照"
            self.model_info_label.SetLabel(self.model_info)
            self.Layout()
            self.update_status()
            wx.MessageBox(
                f"在 {os.path.abspath(SAVE_ROOT)} 下没有找到 "
                f"<局数>/{SUB_DIR}/ 目录。\n请先运行训练脚本产生存档。",
                "未找到存档", wx.OK | wx.ICON_WARNING)

    # ---------------- 对局控制 ----------------
    def new_game(self):
        mode = self.mode_combo.GetValue()
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
        self._ai_prediction = None

        self.board_panel.Refresh()
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
        self.board_panel.Refresh()
        self.update_status()

    def handle_click(self, x, y):
        if self.busy or self.winner is not None:
            return
        if self.human_color is None or self.current != self.human_color:
            return
        j = int(round((x - MARGIN) / CELL))
        i = int(round((y - MARGIN) / CELL))
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
        self.board_panel.Refresh()

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

    # ---------------- AI ----------------
    def maybe_ai_move(self):
        if self.winner is not None or self.busy:
            return
        if self.human_color is not None and self.current == self.human_color:
            return
        self.busy = True
        gid = self.game_id
        wx.CallLater(60, self._do_ai_move, gid)

    def _do_ai_move(self, gid):
        if gid != self.game_id or self.winner is not None:
            self.busy = False
            return
        agent = self.agent_black if self.current == 1 else self.agent_white
        if not agent.loaded:
            self.busy = False
            wx.MessageBox("还没有载入 AI 权重，请先选择快照并点击【载入】。",
                          "未载入模型", wx.OK | wx.ICON_WARNING)
            return
        valid = self.get_valid_actions_idx()
        if not valid:
            self.busy = False
            return
        idx = agent.act(self.get_state(), valid)
        i, j = divmod(idx, BOARD_SIZE)
        if self.board[i][j] != 0:
            i, j = divmod(valid[0], BOARD_SIZE)
        self.busy = False
        self.place(i, j)

    # ---------------- 规则 / 状态 ----------------
    def get_state(self):
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

    # ---------------- AI 思考 ----------------
    def _compute_ai_view(self):
        if self.winner is not None or self.busy:
            return None
        agent = self.agent_black if self.current == 1 else self.agent_white
        if not agent.loaded:
            return None
        valid = self.get_valid_actions_idx()
        if not valid:
            return None
        q = agent.qvalues(self.get_state(), valid)
        return q, valid

    # ---------------- 威胁检测 ----------------
    def detect_threats(self):
        n = BOARD_SIZE
        board = self.board
        threats = {}
        seen = set()

        for i in range(n):
            for j in range(n):
                p = board[i][j]
                if p == 0:
                    continue
                for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    pi, pj = i - dx, j - dy
                    if 0 <= pi < n and 0 <= pj < n and board[pi][pj] == p:
                        continue
                    length = 0
                    cells = []
                    x, y = i, j
                    while 0 <= x < n and 0 <= y < n and board[x][y] == p:
                        cells.append((x, y))
                        length += 1
                        x += dx
                        y += dy
                    if length < 3:
                        continue
                    open_ends = 0
                    if (0 <= i - dx < n and 0 <= j - dy < n
                            and board[i - dx][j - dy] == 0):
                        open_ends += 1
                    if 0 <= x < n and 0 <= y < n and board[x][y] == 0:
                        open_ends += 1

                    if length >= 5:
                        ttype = 'five'
                    elif length == 4 and open_ends == 2:
                        ttype = 'live_four'
                    elif length == 4 and open_ends == 1:
                        ttype = 'rush_four'
                    elif length == 3 and open_ends == 2:
                        ttype = 'live_three'
                    elif length == 3 and open_ends == 1:
                        ttype = 'sleep_three'
                    else:
                        continue

                    key = (tuple(cells), dx, dy)
                    if key in seen:
                        continue
                    seen.add(key)

                    for c in cells:
                        cur = threats.get(c)
                        if cur is None or THREAT_PRIORITY[ttype] > THREAT_PRIORITY[cur]:
                            threats[c] = ttype
        return threats

    # ---------------- 绘制 ----------------
    def draw_board(self, gc):
        ai_view = self._compute_ai_view()

        if ai_view and self.cb_heat.GetValue():
            q, valid = ai_view
            self._draw_heatmap(gc, q, valid)

        self._draw_grid(gc)
        self._draw_stones(gc)

        if self.cb_threat.GetValue():
            self._draw_threats(gc)

        self._draw_last_move(gc)

        self._ai_prediction = None
        if ai_view:
            q, valid = ai_view
            top = max(valid, key=lambda v: q[v])
            self._ai_prediction = divmod(top, BOARD_SIZE)
            self.update_status()

    def _draw_grid(self, gc):
        n = BOARD_SIZE
        x0 = y0 = MARGIN
        x1 = y1 = MARGIN + CELL * (n - 1)
        gc.SetPen(wx.Pen(wx.Colour(GRID_HEX), 1))
        for k in range(n):
            gc.StrokeLine(x0, y0 + k * CELL, x1, y0 + k * CELL)
            gc.StrokeLine(x0 + k * CELL, y0, x0 + k * CELL, y1)
        gc.SetBrush(wx.Brush(wx.Colour(GRID_HEX)))
        gc.SetPen(wx.TRANSPARENT_PEN)
        for si in (3, 6, 9):
            for sj in (3, 6, 9):
                cx = MARGIN + sj * CELL
                cy = MARGIN + si * CELL
                gc.DrawEllipse(cx - 3, cy - 3, 6, 6)

    def _draw_heatmap(self, gc, q, valid):
        if not valid:
            return
        valid_q = q[valid]
        q_min = float(valid_q.min())
        q_max = float(valid_q.max())
        span = q_max - q_min
        r = CELL * 0.44
        gc.SetPen(wx.TRANSPARENT_PEN)
        for idx in valid:
            v = 0.5 if span < 1e-6 else (float(q[idx]) - q_min) / span
            i, j = divmod(idx, BOARD_SIZE)
            cx = MARGIN + j * CELL
            cy = MARGIN + i * CELL
            rgb = heat_fill_rgb(v)
            gc.SetBrush(wx.Brush(wx.Colour(*rgb)))
            gc.DrawEllipse(cx - r, cy - r, 2 * r, 2 * r)

    def _draw_stones(self, gc):
        r = CELL * 0.42
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                p = self.board[i][j]
                if p == 0:
                    continue
                cx = MARGIN + j * CELL
                cy = MARGIN + i * CELL
                if p == 1:
                    gc.SetBrush(wx.Brush(wx.Colour(26, 26, 26)))
                    gc.SetPen(wx.Pen(wx.Colour(0, 0, 0), 1))
                else:
                    gc.SetBrush(wx.Brush(wx.Colour(250, 250, 250)))
                    gc.SetPen(wx.Pen(wx.Colour(136, 136, 136), 1))
                gc.DrawEllipse(cx - r, cy - r, 2 * r, 2 * r)

    def _draw_threats(self, gc):
        threats = self.detect_threats()
        if not threats:
            return
        r_col = CELL * 0.48
        r_out = r_col + 3
        gc.SetBrush(wx.TRANSPARENT_BRUSH)
        for (i, j), ttype in threats.items():
            cx = MARGIN + j * CELL
            cy = MARGIN + i * CELL
            color = THREAT_COLORS.get(ttype, '#FFFFFF')
            gc.SetPen(wx.Pen(wx.Colour(255, 255, 255), 2))
            gc.DrawEllipse(cx - r_out, cy - r_out, 2 * r_out, 2 * r_out)
            gc.SetPen(wx.Pen(wx.Colour(color), 5))
            gc.DrawEllipse(cx - r_col, cy - r_col, 2 * r_col, 2 * r_col)

    def _draw_last_move(self, gc):
        if not self.last_move:
            return
        i, j = self.last_move
        cx = MARGIN + j * CELL
        cy = MARGIN + i * CELL
        gc.SetBrush(wx.TRANSPARENT_BRUSH)
        gc.SetPen(wx.Pen(wx.Colour(229, 57, 53), 2))
        gc.DrawEllipse(cx - 5, cy - 5, 10, 10)

    # ---------------- 状态栏 ----------------
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

        pred = ""
        if self._ai_prediction:
            ti, tj = self._ai_prediction
            pred = f"    |    AI 推荐: ({ti}, {tj})"

        self.status_text.SetLabel(state + pred)
        self.status_text.Refresh()


# ------------------------------------------------------------
class GoBangApp(wx.App):
    def OnInit(self):
        frame = GoBangFrame()
        frame.Show()
        return True


def main():
    app = GoBangApp(False)
    app.MainLoop()


if __name__ == "__main__":
    main()
