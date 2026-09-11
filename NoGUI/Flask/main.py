# app.py —— Flask 版五子棋 DQN 自对弈训练（无访问也自动训练）
import os
import json
import time
import random
import shutil
import threading
import traceback
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from flask import Flask, jsonify, render_template_string

# ===================== 参数 =====================
BOARD_SIZE = 13
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
STATE_SIZE = ACTION_SIZE

REGION_REWARD = [
    [0.05, 0.02, 0.05],
    [0.02, 0.10, 0.02],
    [0.05, 0.02, 0.05],
]
SHAPE_REWARD = {
    'live_four': 0.5, 'rush_four': 0.2, 'live_three': 0.1,
    'sleep_three': 0.02, 'live_two': 0.01,
}
THREAT_PENALTY = {
    'live_four': -0.5, 'rush_four': -0.2, 'live_three': -0.1,
}

LR = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_CAPACITY = 20000
TARGET_UPDATE_INTERVAL = 10
AUTO_SAVE_INTERVAL = 500          # 每多少局自动存一次档
MAX_SAVES = 20                    # 最多保留 20 份存档，超出覆盖最旧的
WEIGHT_ROOT = "json/bogang"
STEP_DELAY = 0.05                 # 每步落子的间隔（秒），纯粹为了网页上看得清；想快就改成 0

os.makedirs(WEIGHT_ROOT, exist_ok=True)
# ================================================


# --------------------------- 游戏环境 ---------------------------
class GoBangGame:
    def __init__(self):
        self.size = BOARD_SIZE
        self.board = None
        self.current_player = None
        self.winner = None
        self.reset()

    def reset(self):
        self.board = [[0] * self.size for _ in range(self.size)]
        self.current_player = 1
        self.winner = None
        return self.get_state()

    def get_state(self):
        s = np.zeros((self.size, self.size), dtype=np.float32)
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 1:
                    s[i][j] = 1
                elif self.board[i][j] == 2:
                    s[i][j] = -1
        return s

    def get_valid_actions_idx(self):
        return [i * self.size + j
                for i in range(self.size)
                for j in range(self.size)
                if self.board[i][j] == 0]

    def get_shape_reward(self, i, j, player):
        reward = 0.0
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            length = 1
            open_ends = 0
            x, y = i + dx, j + dy
            while 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == player:
                length += 1; x += dx; y += dy
            if 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == 0:
                open_ends += 1
            x, y = i - dx, j - dy
            while 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == player:
                length += 1; x -= dx; y -= dy
            if 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == 0:
                open_ends += 1

            if length >= 5:
                continue
            elif length == 4:
                if open_ends == 2: reward += SHAPE_REWARD['live_four']
                elif open_ends == 1: reward += SHAPE_REWARD['rush_four']
            elif length == 3:
                if open_ends == 2: reward += SHAPE_REWARD['live_three']
                elif open_ends == 1: reward += SHAPE_REWARD['sleep_three']
            elif length == 2 and open_ends == 2:
                reward += SHAPE_REWARD['live_two']
        return reward

    def get_opponent_threat_penalty(self, opponent):
        penalty = 0.0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != opponent:
                    continue
                for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                    x, y = i - dx, j - dy
                    if 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == opponent:
                        continue
                    length = 0
                    x, y = i, j
                    while 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == opponent:
                        length += 1; x += dx; y += dy
                    open_ends = 0
                    x1, y1 = i - dx, j - dy
                    if 0 <= x1 < self.size and 0 <= y1 < self.size and self.board[x1][y1] == 0:
                        open_ends += 1
                    x2, y2 = i + (length - 1) * dx, j + (length - 1) * dy
                    x2 += dx; y2 += dy
                    if 0 <= x2 < self.size and 0 <= y2 < self.size and self.board[x2][y2] == 0:
                        open_ends += 1
                    if length >= 5:
                        continue
                    if length == 4:
                        if open_ends == 2: penalty += THREAT_PENALTY['live_four']
                        elif open_ends == 1: penalty += THREAT_PENALTY['rush_four']
                    elif length == 3 and open_ends == 2:
                        penalty += THREAT_PENALTY['live_three']
        return penalty

    def make_move(self, action_idx):
        i = action_idx // self.size
        j = action_idx % self.size
        if self.board[i][j] != 0:
            return 0.0, False
        self.board[i][j] = self.current_player

        region_i = min(i // (self.size // 3), 2)
        region_j = min(j // (self.size // 3), 2)
        region_reward = REGION_REWARD[region_i][region_j]
        shape_reward = self.get_shape_reward(i, j, self.current_player)
        opponent = 3 - self.current_player
        threat_penalty = self.get_opponent_threat_penalty(opponent)
        total_reward = region_reward + shape_reward + threat_penalty

        done = False
        if self.check_winner(i, j):
            self.winner = self.current_player
            done = True
        elif len(self.get_valid_actions_idx()) == 0:
            self.winner = 0
            done = True
        else:
            self.current_player = opponent
        return total_reward, done

    def check_winner(self, i, j):
        player = self.board[i][j]
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            count = 1
            for step in (1, -1):
                x, y = i + dx * step, j + dy * step
                while 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == player:
                    count += 1; x += dx * step; y += dy * step
            if count >= 5:
                return True
        return False


# --------------------------- DQN ---------------------------
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, ACTION_SIZE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor(np.array([e[0].flatten() for e in batch]))
        actions = torch.LongTensor(np.array([e[1] for e in batch]))
        rewards = torch.FloatTensor(np.array([e[2] for e in batch]))
        next_states = torch.FloatTensor(np.array([e[3].flatten() for e in batch]))
        dones = torch.BoolTensor(np.array([e[4] for e in batch]))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.target = DQN().to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        self.epsilon = EPSILON_START

    def act(self, state, valid_actions):
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            st = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
            q = self.model(st).cpu().numpy().flatten()
        valid_set = set(valid_actions)
        for idx in range(ACTION_SIZE):
            if idx not in valid_set:
                q[idx] = -np.inf
        return int(np.argmax(q))

    def remember(self, s, a, r, ns, d):
        self.memory.push(s, a, r, ns, d)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).max(1)[0]
            target_q = rewards + (1 - dones.float()) * GAMMA * next_q
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
        return loss.item()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    # ---- 权重用 torch.save 保存（比 JSON 快很多，体积也小很多）----
    def save(self, filepath):
        torch.save({
            'model': self.model.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)

    def load(self, filepath):
        if not os.path.exists(filepath):
            return False
        try:
            ckpt = torch.load(filepath, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.target.load_state_dict(ckpt['model'])
        self.epsilon = ckpt.get('epsilon', EPSILON_START)
        return True


# --------------------------- 存档管理（最多 20 份）---------------------------
def list_saves():
    """返回按局数从小到大排序的存档目录名列表。"""
    if not os.path.isdir(WEIGHT_ROOT):
        return []
    dirs = [d for d in os.listdir(WEIGHT_ROOT)
            if d.isdigit() and os.path.isdir(os.path.join(WEIGHT_ROOT, d))]
    return sorted(dirs, key=int)


def save_checkpoint(agent_black, agent_white, episode):
    save_dir = os.path.join(WEIGHT_ROOT, str(episode))
    os.makedirs(save_dir, exist_ok=True)
    agent_black.save(os.path.join(save_dir, "black.pt"))
    agent_white.save(os.path.join(save_dir, "white.pt"))
    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "episode": episode,
            "epsilon_black": agent_black.epsilon,
            "epsilon_white": agent_white.epsilon,
        }, f, ensure_ascii=False, indent=2)

    # 超过 20 份就删掉最旧的
    saves = list_saves()
    while len(saves) > MAX_SAVES:
        oldest = saves.pop(0)
        shutil.rmtree(os.path.join(WEIGHT_ROOT, oldest), ignore_errors=True)
        print(f"  → 清理旧存档 {oldest}（最多保留 {MAX_SAVES} 份）")
    return save_dir


# --------------------------- 后台训练线程 ---------------------------
class Trainer(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.game = GoBangGame()
        self.agent_black = DQNAgent()
        self.agent_white = DQNAgent()

        self.episode = 0
        self.move_count = 0
        self.last_move = None
        self.last_loss = 0.0
        self.black_wins = 0
        self.white_wins = 0
        self.draws = 0

        self.snapshot = {}
        self._load_latest()
        self._publish()

    # ---------- 存档 ----------
    def _load_latest(self):
        saves = list_saves()
        if not saves:
            return
        latest = saves[-1]
        d = os.path.join(WEIGHT_ROOT, latest)
        b = os.path.join(d, "black.pt")
        w = os.path.join(d, "white.pt")
        try:
            if os.path.exists(b) and os.path.exists(w):
                self.agent_black.load(b)
                self.agent_white.load(w)
                self.episode = int(latest)
                print(f"已加载存档 {latest}，从第 {self.episode} 局继续训练")
        except Exception as e:
            print("加载存档失败，从头开始：", e)

    # ---------- 对外快照 ----------
    def _publish(self):
        snap = {
            "episode": self.episode,
            "board": [row[:] for row in self.game.board],
            "last_move": list(self.last_move) if self.last_move else None,
            "moves": self.move_count,
            "winner": self.game.winner,
            "stats": {
                "black_wins": self.black_wins,
                "white_wins": self.white_wins,
                "draws": self.draws,
                "epsilon_black": round(self.agent_black.epsilon, 4),
                "epsilon_white": round(self.agent_white.epsilon, 4),
                "loss": round(self.last_loss, 4),
                "memory": len(self.agent_black.memory),
                "saves": len(list_saves()),
            },
        }
        with self.lock:
            self.snapshot = snap

    def get_snapshot(self):
        with self.lock:
            return self.snapshot

    # ---------- 主循环 ----------
    def run(self):
        print("后台训练线程已启动（无访问也会持续训练）...")
        while True:
            try:
                self.play_episode()
            except Exception:
                traceback.print_exc()
                time.sleep(1)

    def play_episode(self):
        game = self.game
        game.reset()
        history = []
        self.move_count = 0
        self.last_move = None
        self._publish()

        while game.winner is None:
            player = game.current_player
            agent = self.agent_black if player == 1 else self.agent_white
            state = game.get_state()
            valid = game.get_valid_actions_idx()
            if not valid:
                game.winner = 0
                break

            action = agent.act(state, valid)
            reward, done = game.make_move(action)
            next_state = game.get_state()
            history.append((player, state.copy(), action, reward, next_state.copy(), done))

            self.move_count += 1
            self.last_move = (action // game.size, action % game.size)
            self._publish()          # 每落一子就刷新网页数据
            if STEP_DELAY:
                time.sleep(STEP_DELAY)

        winner = game.winner

        # 结算：获胜方最后一步 +1
        for p, s, a, r, ns, d in history:
            total = r + (1.0 if d and winner == p else 0.0)
            if p == 1:
                self.agent_black.remember(s, a, total, ns, d)
            else:
                self.agent_white.remember(s, a, total, ns, d)

        # 真训练
        loss_b = self.agent_black.replay()
        loss_w = self.agent_white.replay()
        losses = [l for l in (loss_b, loss_w) if l is not None]
        if losses:
            self.last_loss = float(np.mean(losses))

        self.episode += 1
        if winner == 1:
            self.black_wins += 1
        elif winner == 2:
            self.white_wins += 1
        else:
            self.draws += 1

        if self.episode % TARGET_UPDATE_INTERVAL == 0:
            self.agent_black.update_target()
            self.agent_white.update_target()

        if self.episode % AUTO_SAVE_INTERVAL == 0:
            d = save_checkpoint(self.agent_black, self.agent_white, self.episode)
            print(f"  → 已保存到 {d}（当前存档数 {len(list_saves())}/{MAX_SAVES}）")

        self._publish()


# --------------------------- Flask ---------------------------
app = Flask(__name__)
trainer = Trainer()
trainer.start()          # 模块加载即启动，无需任何访问

INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>五子棋 DQN 自对弈训练</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px 16px 48px;
    background: radial-gradient(circle at 50% 0%, #1c2029, #0f1116 70%);
    color: #e6e8eb; min-height: 100vh;
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    display: flex; flex-direction: column; align-items: center;
  }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 6px; letter-spacing: .5px; }
  .sub { color: #7d8695; font-size: 13px; margin-bottom: 20px; }
  .wrap { display: flex; gap: 28px; align-items: flex-start; flex-wrap: wrap; justify-content: center; }
  .board {
    display: grid;
    width: min(560px, 90vw);
    aspect-ratio: 1 / 1;
    background: #e3b877;
    border: 8px solid #8a5f2b;
    border-radius: 10px;
    box-shadow: 0 14px 40px rgba(0,0,0,.6), inset 0 0 40px rgba(120,80,20,.35);
    overflow: hidden;
  }
  .cell { position: relative; border: 1px solid rgba(90,60,20,.25); }
  .cell.black::after, .cell.white::after {
    content: ''; position: absolute; inset: 10%; border-radius: 50%;
  }
  .cell.black::after { background: radial-gradient(circle at 34% 30%, #7a7a7a, #000 72%); }
  .cell.white::after { background: radial-gradient(circle at 34% 30%, #ffffff, #b6b6b6 78%); }
  .cell.last::before {
    content: ''; position: absolute; inset: 3%; border-radius: 50%;
    border: 2px solid #e03131; box-shadow: 0 0 10px rgba(224,49,49,.8);
  }
  .panel {
    width: 300px; background: #171a21; border: 1px solid #23262d;
    border-radius: 12px; padding: 16px 18px;
    box-shadow: 0 14px 40px rgba(0,0,0,.45);
  }
  .panel h2 { font-size: 14px; margin: 0 0 12px; color: #8b93a1; font-weight: 600; letter-spacing: 1px; }
  .row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid #23262d; font-size: 14px;
  }
  .row:last-child { border-bottom: none; }
  .k { color: #8b93a1; }
  .v { font-weight: 600; font-variant-numeric: tabular-nums; }
  .badge { padding: 2px 10px; border-radius: 999px; font-size: 12px; }
  .b-live { background: #1f3a2a; color: #5fe08a; }
  .b-done { background: #3a1f22; color: #ff8a8a; }
</style>
</head>
<body>
  <h1>五子棋 DQN 自对弈训练</h1>
  <div class="sub">后台自动训练中 · 关闭页面也不会停止 · 最多保留 20 份存档</div>

  <div class="wrap">
    <div class="board" id="board"></div>

    <div class="panel">
      <h2>训练状态</h2>
      <div class="row"><span class="k">当前局数</span><span class="v" id="episode">0</span></div>
      <div class="row"><span class="k">本局步数</span><span class="v" id="moves">0</span></div>
      <div class="row"><span class="k">对局状态</span><span class="v" id="status">进行中</span></div>
      <div class="row"><span class="k">黑胜 / 白胜 / 平</span><span class="v" id="record">0 / 0 / 0</span></div>
      <div class="row"><span class="k">ε 黑 / 白</span><span class="v" id="eps">-</span></div>
      <div class="row"><span class="k">最近 Loss</span><span class="v" id="loss">-</span></div>
      <div class="row"><span class="k">经验池</span><span class="v" id="mem">0</span></div>
      <div class="row"><span class="k">已保存存档</span><span class="v" id="saves">0 / 20</span></div>
    </div>
  </div>

<script>
const N = 13;
const boardEl = document.getElementById('board');
const cells = [];
for (let i = 0; i < N; i++) {
  cells[i] = [];
  for (let j = 0; j < N; j++) {
    const d = document.createElement('div');
    d.className = 'cell';
    boardEl.appendChild(d);
    cells[i][j] = d;
  }
}

const $ = id => document.getElementById(id);

function render(s) {
  $('episode').textContent = s.episode;
  $('moves').textContent = s.moves;
  $('record').textContent = `${s.stats.black_wins} / ${s.stats.white_wins} / ${s.stats.draws}`;
  $('eps').textContent = s.stats.epsilon_black.toFixed(3) + ' / ' + s.stats.epsilon_white.toFixed(3);
  $('loss').textContent = s.stats.loss.toFixed(4);
  $('mem').textContent = s.stats.memory;
  $('saves').textContent = s.stats.saves + ' / 20';

  const st = $('status');
  if (s.winner === 1)       { st.textContent = '黑胜'; st.className = 'v badge b-done'; }
  else if (s.winner === 2)  { st.textContent = '白胜'; st.className = 'v badge b-done'; }
  else if (s.winner === 0)  { st.textContent = '平局'; st.className = 'v badge b-done'; }
  else                      { st.textContent = '进行中'; st.className = 'v badge b-live'; }

  const lm = s.last_move, b = s.board;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const v = b[i][j];
      let cls = 'cell';
      if (v === 1) cls += ' black';
      else if (v === 2) cls += ' white';
      if (lm && lm[0] === i && lm[1] === j) cls += ' last';
      const c = cells[i][j];
      if (c.className !== cls) c.className = cls;
    }
  }
}

async function poll() {
  try {
    const r = await fetch('/api/state', { cache: 'no-store' });
    if (r.ok) render(await r.json());
  } catch (e) { /* 忽略网络抖动 */ }
  setTimeout(poll, 120);
}
poll();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/state")
def api_state():
    return jsonify(trainer.get_snapshot())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
