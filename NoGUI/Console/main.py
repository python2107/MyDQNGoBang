# train.py —— 五子棋 DQN 自对弈训练（纯训练版 · JSON 存档）
#
# 存档格式：《五子棋 DQN JSON 存档格式规范 v1》
# 保存路径：json/gobang/{episode}/qgnn13/
#     black.json   黑方智能体（fmt/board/hidden/epsilon/updates/model）
#     white.json   白方智能体
#     meta.json    训练进度（fmt/board/hidden/episode/b_wins/w_wins/draws）
# 权重按 PyTorch 原生 (out, in) 布局存储，ESP32 端加载时自行转置。
#
# 用法：
#   python train.py                     # 无限训练，Ctrl+C 退出并自动存档
#   python train.py --episodes 5000     # 训练 5000 局后退出
#   python train.py --log-interval 20   # 每 20 局打印一次日志
#   python train.py --board-size 13 --hidden 256 --seed 42
#
import os
import json
import time
import random
import shutil
import argparse
import traceback
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ===================== 参数 =====================
FMT          = 1                    # JSON 存档格式版本号
BOARD_SIZE   = 13
ACTION_SIZE  = BOARD_SIZE * BOARD_SIZE
STATE_SIZE   = ACTION_SIZE
HIDDEN       = 256                  # 隐藏层宽度

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
AUTO_SAVE_INTERVAL = 500
MAX_SAVES = 20
WEIGHT_ROOT = "json/gobang"     # 存档根目录
SAVE_SUBDIR = "qgnn13"          # 每个存档下固定的子目录
LOG_INTERVAL = 10

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
        self.fc1 = nn.Linear(STATE_SIZE, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, ACTION_SIZE)

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
        self.updates = 0

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

        self.updates += 1
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
        if self.updates % TARGET_UPDATE_INTERVAL == 0:
            self.update_target()
        return loss.item()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    # ---- JSON 存档 ----
    def save_json(self, path):
        state = {k: v.detach().cpu().numpy().tolist()
                 for k, v in self.model.state_dict().items()}
        payload = {
            "fmt":     FMT,
            "board":   BOARD_SIZE,
            "hidden":  HIDDEN,
            "epsilon": float(self.epsilon),
            "updates": int(self.updates),
            "model":   state,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load_json(self, path):
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print(f"  读取 {path} 失败：{e}")
            return False

        if d.get("fmt") != FMT:
            print(f"  {os.path.basename(path)} fmt={d.get('fmt')} 不匹配，跳过")
            return False
        if d.get("board") != BOARD_SIZE:
            print(f"  {os.path.basename(path)} board={d.get('board')} "
                  f"与当前 BOARD_SIZE={BOARD_SIZE} 不一致，跳过")
            return False
        if d.get("hidden") != HIDDEN:
            print(f"  {os.path.basename(path)} hidden={d.get('hidden')} "
                  f"与当前 HIDDEN={HIDDEN} 不一致，跳过")
            return False

        try:
            state = {k: torch.tensor(v, dtype=torch.float32)
                     for k, v in d["model"].items()}
        except Exception as e:
            print(f"  {os.path.basename(path)} model 字段解析失败：{e}")
            return False

        self.model.load_state_dict(state)
        self.target.load_state_dict(state)
        self.epsilon = float(d.get("epsilon", EPSILON_START))
        self.updates = int(d.get("updates", 0))
        return True


# --------------------------- 存档管理 ---------------------------
def list_saves():
    """返回按局数排序的有效存档目录名（要求含完整 qgnn13/black.json 等）。"""
    if not os.path.isdir(WEIGHT_ROOT):
        return []
    dirs = []
    for d in os.listdir(WEIGHT_ROOT):
        if not d.isdigit():
            continue
        full = os.path.join(WEIGHT_ROOT, d, SAVE_SUBDIR)
        if (os.path.isdir(full)
                and os.path.exists(os.path.join(full, "black.json"))
                and os.path.exists(os.path.join(full, "white.json"))):
            dirs.append(d)
    return sorted(dirs, key=int)


def save_checkpoint(agent_black, agent_white, episode,
                    b_wins, w_wins, draws):
    """
    按 JSON 规范写出三个文件。
    路径：json/gobang/{episode}/qgnn13/
        black.json / white.json / meta.json
    """
    save_dir = os.path.join(WEIGHT_ROOT, str(episode), SAVE_SUBDIR)
    os.makedirs(save_dir, exist_ok=True)

    agent_black.save_json(os.path.join(save_dir, "black.json"))
    agent_white.save_json(os.path.join(save_dir, "white.json"))

    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "fmt":     FMT,
            "board":   BOARD_SIZE,
            "hidden":  HIDDEN,
            "episode": int(episode),
            "b_wins":  int(b_wins),
            "w_wins":  int(w_wins),
            "draws":   int(draws),
        }, f)

    saves = list_saves()
    while len(saves) > MAX_SAVES:
        oldest = saves.pop(0)
        shutil.rmtree(os.path.join(WEIGHT_ROOT, oldest), ignore_errors=True)
        print(f"  → 清理旧存档 {oldest}（最多保留 {MAX_SAVES} 份）")
    return save_dir


# --------------------------- 训练器 ---------------------------
class Trainer:
    def __init__(self):
        self.game = GoBangGame()
        self.agent_black = DQNAgent()
        self.agent_white = DQNAgent()

        self.episode = 0
        self.move_count = 0
        self.last_loss = 0.0
        self.black_wins = 0
        self.white_wins = 0
        self.draws = 0

        self._load_latest()

    def _load_latest(self):
        saves = list_saves()
        if not saves:
            return
        latest = saves[-1]
        d = os.path.join(WEIGHT_ROOT, latest, SAVE_SUBDIR)
        b = os.path.join(d, "black.json")
        w = os.path.join(d, "white.json")
        m = os.path.join(d, "meta.json")
        try:
            ok_b = self.agent_black.load_json(b)
            ok_w = self.agent_white.load_json(w)
            if not (ok_b and ok_w):
                print(f"存档 {latest} 加载失败，从头开始")
                return

            self.episode = int(latest)
            if os.path.exists(m):
                try:
                    with open(m, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    self.black_wins = int(meta.get("b_wins", 0))
                    self.white_wins = int(meta.get("w_wins", 0))
                    self.draws      = int(meta.get("draws",  0))
                except Exception as e:
                    print(f"  读取 meta.json 失败：{e}")

            print(f"已加载存档 {latest}（第 {self.episode} 局），"
                  f"黑胜 {self.black_wins} / 白胜 {self.white_wins} / 平 {self.draws}")
        except Exception as e:
            print("加载存档失败，从头开始：", e)

    def play_episode(self):
        game = self.game
        game.reset()
        history = []
        self.move_count = 0

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
            history.append((player, state.copy(), action, reward,
                            next_state.copy(), done))
            self.move_count += 1

        winner = game.winner

        for p, s, a, r, ns, d in history:
            total = r + (1.0 if d and winner == p else 0.0)
            if p == 1:
                self.agent_black.remember(s, a, total, ns, d)
            else:
                self.agent_white.remember(s, a, total, ns, d)

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

        if self.episode % AUTO_SAVE_INTERVAL == 0:
            d = save_checkpoint(self.agent_black, self.agent_white,
                                self.episode,
                                self.black_wins, self.white_wins, self.draws)
            print(f"  → 已保存到 {d}（当前存档数 {len(list_saves())}/{MAX_SAVES}）")

    def log(self, session_start_episode, session_start_time):
        done_eps = max(self.episode - session_start_episode, 1)
        elapsed = time.time() - session_start_time
        eps = (self.agent_black.epsilon + self.agent_white.epsilon) / 2
        total = max(self.black_wins + self.white_wins + self.draws, 1)
        print(
            f"[第 {self.episode:>7} 局] "
            f"黑胜 {self.black_wins:>6} ({self.black_wins/total*100:4.1f}%) / "
            f"白胜 {self.white_wins:>6} ({self.white_wins/total*100:4.1f}%) / "
            f"平 {self.draws:>5} | "
            f"ε {eps:.4f} | loss {self.last_loss:.4f} | "
            f"经验池 {len(self.agent_black.memory):>5} | "
            f"{elapsed/done_eps:.3f}s/局"
        )

    def run(self, total_episodes=0, log_interval=LOG_INTERVAL):
        device = self.agent_black.device
        print("=" * 78)
        print("五子棋 DQN 自对弈训练（纯训练版 · JSON 存档 v1）")
        print(f"设备: {device} | 棋盘: {BOARD_SIZE}x{BOARD_SIZE} | "
              f"HIDDEN: {HIDDEN} | 起始局数: {self.episode}")
        print(f"存档根目录: {os.path.abspath(WEIGHT_ROOT)}")
        print(f"单档结构: {WEIGHT_ROOT}/{{episode}}/{SAVE_SUBDIR}/"
              f"{{black.json, white.json, meta.json}}")
        if total_episodes:
            print(f"计划训练 {total_episodes} 局后退出")
        else:
            print("无限训练模式，按 Ctrl+C 可随时退出（退出前会自动存档）")
        print("=" * 78)

        session_start_episode = self.episode
        session_start_time = time.time()

        try:
            while total_episodes == 0 or self.episode < total_episodes:
                self.play_episode()
                if self.episode % log_interval == 0:
                    self.log(session_start_episode, session_start_time)
        except KeyboardInterrupt:
            print("\n检测到 Ctrl+C，正在保存存档...")
        except Exception:
            traceback.print_exc()
            print("训练异常中断，正在保存存档...")
        finally:
            d = save_checkpoint(self.agent_black, self.agent_white,
                                self.episode,
                                self.black_wins, self.white_wins, self.draws)
            print(f"已保存到 {d}（第 {self.episode} 局），"
                  f"当前存档数 {len(list_saves())}/{MAX_SAVES}")


# --------------------------- 入口 ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="五子棋 DQN 自对弈训练（纯训练版 · JSON 存档 v1）")
    p.add_argument("--episodes", type=int, default=0,
                   help="训练总局数，0 表示无限训练（默认 0）")
    p.add_argument("--log-interval", type=int, default=LOG_INTERVAL,
                   help=f"每多少局打印一次日志（默认 {LOG_INTERVAL}）")
    p.add_argument("--board-size", type=int, default=BOARD_SIZE,
                   help=f"棋盘边长（默认 {BOARD_SIZE}）")
    p.add_argument("--hidden", type=int, default=HIDDEN,
                   help=f"隐藏层宽度（默认 {HIDDEN}）")
    p.add_argument("--seed", type=int, default=None, help="随机种子（可选）")
    p.add_argument("--weight-root", type=str, default=WEIGHT_ROOT,
                   help=f"存档根目录（默认 {WEIGHT_ROOT}）")
    p.add_argument("--save-subdir", type=str, default=SAVE_SUBDIR,
                   help=f"每个存档下的固定子目录（默认 {SAVE_SUBDIR}）")
    return p.parse_args()


def main():
    args = parse_args()

    global BOARD_SIZE, ACTION_SIZE, STATE_SIZE, HIDDEN
    global WEIGHT_ROOT, SAVE_SUBDIR
    BOARD_SIZE = args.board_size
    ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
    STATE_SIZE = ACTION_SIZE
    HIDDEN = args.hidden
    WEIGHT_ROOT = args.weight_root
    SAVE_SUBDIR = args.save_subdir
    os.makedirs(WEIGHT_ROOT, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer()
    trainer.run(total_episodes=args.episodes, log_interval=args.log_interval)


if __name__ == "__main__":
    main()
