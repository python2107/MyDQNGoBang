# train.py —— 五子棋 DQN 自对弈训练（纯训练版，无 Web 界面）
#
# 用法：
#   python train.py                     # 无限训练，Ctrl+C 退出并自动存档
#   python train.py --episodes 5000     # 训练 5000 局后退出
#   python train.py --log-interval 20   # 每 20 局打印一次日志
#   python train.py --board-size 13 --seed 42
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
LOG_INTERVAL = 10                 # 每多少局打印一次训练日志

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

    # ---------- 单局自对弈 ----------
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
            history.append((player, state.copy(), action, reward, next_state.copy(), done))
            self.move_count += 1

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

    # ---------- 日志 ----------
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

    # ---------- 主循环 ----------
    def run(self, total_episodes=0, log_interval=LOG_INTERVAL):
        device = self.agent_black.device
        print("=" * 78)
        print(f"五子棋 DQN 自对弈训练（纯训练版）")
        print(f"设备: {device} | 棋盘: {BOARD_SIZE}x{BOARD_SIZE} | 起始局数: {self.episode}")
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
            d = save_checkpoint(self.agent_black, self.agent_white, self.episode)
            print(f"已保存到 {d}（第 {self.episode} 局），当前存档数 {len(list_saves())}/{MAX_SAVES}")


# --------------------------- 入口 ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="五子棋 DQN 自对弈训练（纯训练版）")
    p.add_argument("--episodes", type=int, default=0,
                   help="训练总局数，0 表示无限训练（默认 0）")
    p.add_argument("--log-interval", type=int, default=LOG_INTERVAL,
                   help=f"每多少局打印一次日志（默认 {LOG_INTERVAL}）")
    p.add_argument("--board-size", type=int, default=BOARD_SIZE,
                   help=f"棋盘边长（默认 {BOARD_SIZE}）")
    p.add_argument("--seed", type=int, default=None, help="随机种子（可选）")
    p.add_argument("--weight-root", type=str, default=WEIGHT_ROOT,
                   help=f"存档根目录（默认 {WEIGHT_ROOT}）")
    return p.parse_args()


def main():
    args = parse_args()

    # 允许通过命令行覆盖全局常量（必须在创建任何 Agent / Game 之前）
    global BOARD_SIZE, ACTION_SIZE, STATE_SIZE, WEIGHT_ROOT
    BOARD_SIZE = args.board_size
    ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
    STATE_SIZE = ACTION_SIZE
    WEIGHT_ROOT = args.weight_root
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
