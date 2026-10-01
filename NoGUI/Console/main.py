# ============================================================
#  五子棋 DQN 训练主程序 (13 路) —— 防守强化版
#  存档路径：json/gobang/{局数}/qgnn13/{black,white,meta}.json
# ============================================================
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# ==================== 全局配置 ====================
BOARD_SIZE   = 13
HIDDEN       = 128 #不能变！！！！！！！！！！！
ACTION_SIZE  = BOARD_SIZE * BOARD_SIZE
STATE_SIZE   = ACTION_SIZE

FMT_VERSION  = 1
SAVE_ROOT    = "json/gobang"
SUB_DIR      = "qgnn13"
AUTO_SAVE_INTERVAL = 500

# 奖励配置
REGION_REWARD = [
    [0.005, 0.002, 0.005],
    [0.002, 0.010, 0.002],
    [0.005, 0.002, 0.005],
]
SHAPE_REWARD = {
    'live_four':   0.05,
    'rush_four':   0.02,
    'live_three':  0.01,
    'sleep_three': 0.002,
    'live_two':    0.001,
}
THREAT_PENALTY = {
    'live_four':  -0.5,   # 保持高额惩罚
    'rush_four':  -0.2,
    'live_three': -0.1,
}

# DQN 超参数（稳定性调优）
LR              = 5e-5   # 降低学习率，防止震荡
GAMMA           = 0.995  # 提高折扣因子，看重长远收益
EPSILON_START   = 1.0
EPSILON_MIN     = 0.05   # 保留 5% 探索
EPSILON_DECAY   = 0.9995 # 衰减更平缓
BATCH_SIZE      = 128    # 加大批次
MEMORY_CAPACITY = 50000  # 扩大经验池，防遗忘
TARGET_UPDATE_INTERVAL = 20 # 延迟目标网络更新

os.makedirs(SAVE_ROOT, exist_ok=True)
# ==================================================


def snapshot_dir(episode):
    return os.path.join(SAVE_ROOT, str(episode), SUB_DIR)


# -------------------- 游戏环境 --------------------
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
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
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

            if length >= 5: continue
            elif length == 4:
                if open_ends == 2:   reward += SHAPE_REWARD['live_four']
                elif open_ends == 1: reward += SHAPE_REWARD['rush_four']
            elif length == 3:
                if open_ends == 2:   reward += SHAPE_REWARD['live_three']
                elif open_ends == 1: reward += SHAPE_REWARD['sleep_three']
            elif length == 2 and open_ends == 2:
                reward += SHAPE_REWARD['live_two']
        return reward

    def get_opponent_threat_penalty(self, opponent):
        """返回对手的威胁惩罚值（负数）"""
        penalty = 0.0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != opponent: continue
                for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
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
                    x2, y2 = i + (length - 1) * dx + dx, j + (length - 1) * dy + dy
                    if 0 <= x2 < self.size and 0 <= y2 < self.size and self.board[x2][y2] == 0:
                        open_ends += 1

                    if length >= 5: continue
                    if length == 4:
                        if open_ends == 2:   penalty += THREAT_PENALTY['live_four']
                        elif open_ends == 1: penalty += THREAT_PENALTY['rush_four']
                    elif length == 3 and open_ends == 2:
                        penalty += THREAT_PENALTY['live_three']
        return penalty

    def make_move(self, action_idx):
        i = action_idx // self.size
        j = action_idx % self.size
        if self.board[i][j] != 0:
            return 0.0, False

        opponent = 3 - self.current_player
        
        # 1. 记录落子前的对手威胁
        threat_before = self.get_opponent_threat_penalty(opponent)
        
        # 2. 执行落子
        self.board[i][j] = self.current_player
        
        # 3. 记录落子后的对手威胁
        threat_after = self.get_opponent_threat_penalty(opponent)
        
        # 4. 计算防守奖励（核心修复！）
        # 如果威胁减少了（threat_before 是负数，threat_after 变大），给正奖励
        defense_reward = (threat_after - threat_before) * 2.0

        region_i = min(i // (self.size // 3), 2)
        region_j = min(j // (self.size // 3), 2)
        region_reward = REGION_REWARD[region_i][region_j]
        shape_reward = self.get_shape_reward(i, j, self.current_player)
        
        # 总奖励 = 位置奖励 + 棋型奖励 + 剩余威胁惩罚 + 防守奖励
        total_reward = region_reward + shape_reward + threat_after + defense_reward

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
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            cnt = 1
            for step in (1, -1):
                x, y = i + dx * step, j + dy * step
                while 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == player:
                    cnt += 1; x += dx * step; y += dy * step
            if cnt >= 5: return True
        return False


# -------------------- DQN 网络 --------------------
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
        states      = torch.FloatTensor(np.array([e[0].flatten() for e in batch]))
        actions     = torch.LongTensor(np.array([e[1] for e in batch]))
        rewards     = torch.FloatTensor(np.array([e[2] for e in batch]))
        next_states = torch.FloatTensor(np.array([e[3].flatten() for e in batch]))
        dones       = torch.BoolTensor(np.array([e[4] for e in batch]))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# -------------------- DQN Agent --------------------
class DQNAgent:
    def __init__(self, name="agent"):
        self.name = name
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
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).max(1)[0]
            target_q = rewards + (1 - dones.float()) * GAMMA * next_q
            target_q = torch.clamp(target_q, -10.0, 10.0) # 防爆

        loss = F.mse_loss(current_q, target_q)
        if loss.item() > 100.0:
            return None

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
        self.updates += 1
        return loss.item()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def save_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {k: v.cpu().numpy().tolist() for k, v in self.model.state_dict().items()}
        payload = {
            "fmt": FMT_VERSION, "board": BOARD_SIZE, "hidden": HIDDEN,
            "epsilon": float(self.epsilon), "updates": int(self.updates),
            "model": state,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"  [保存] {path} (ε={self.epsilon:.4f}, updates={self.updates})")

    def load_json(self, path):
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if d.get("fmt") != FMT_VERSION or d.get("board") != BOARD_SIZE or d.get("hidden") != HIDDEN:
                print(f"  [加载] 格式不匹配，跳过。")
                return False
            state = {k: torch.tensor(v) for k, v in d["model"].items()}
            self.model.load_state_dict(state)
            self.target.load_state_dict(state)
            self.epsilon = float(d.get("epsilon", EPSILON_START))
            self.updates = int(d.get("updates", 0))
            return True
        except Exception as e:
            print(f"  [加载] 失败: {e}")
            return False


# -------------------- 训练流程 --------------------
def save_meta(path, episode, b_wins, w_wins, draws):
    payload = {
        "fmt": FMT_VERSION, "board": BOARD_SIZE, "hidden": HIDDEN,
        "episode": int(episode), "b_wins": int(b_wins), "w_wins": int(w_wins), "draws": int(draws)
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

def load_meta(path):
    if not os.path.exists(path): return None
    try:
        with open(path, "r", encoding="utf-8") as f: d = json.load(f)
        if d.get("fmt") != FMT_VERSION or d.get("board") != BOARD_SIZE: return None
        return d
    except: return None

def find_latest_snapshot():
    if not os.path.isdir(SAVE_ROOT): return 0
    eps = [int(d) for d in os.listdir(SAVE_ROOT) if d.isdigit() and os.path.isdir(os.path.join(SAVE_ROOT, d))]
    return max(eps) if eps else 0

def train_step(agent_black, agent_white, game):
    game.reset()
    history = []
    moves = 0
    while game.winner is None:
        player = game.current_player
        agent = agent_black if player == 1 else agent_white
        state = game.get_state()
        valid = game.get_valid_actions_idx()
        if not valid:
            game.winner = 0
            break
        action = agent.act(state, valid)
        reward, done = game.make_move(action)
        next_state = game.get_state()
        history.append((player, state.copy(), action, reward, next_state.copy(), done))
        moves += 1

    winner = game.winner
    # 核心修复：败方最后一步重罚 -0.5，胜方 +1.0
    for p, s, a, r, ns, d in history:
        if d:
            if winner == p: total = r + 1.0
            elif winner == 3 - p: total = r - 0.5
            else: total = r # 平局
        else:
            total = r
            
        if p == 1: agent_black.remember(s, a, total, ns, d)
        else: agent_white.remember(s, a, total, ns, d)

    lb = agent_black.replay()
    lw = agent_white.replay()
    losses = [l for l in (lb, lw) if l is not None]
    return winner, moves, float(np.mean(losses)) if losses else 0.0

def main():
    game = GoBangGame()
    agent_black = DQNAgent("black")
    agent_white = DQNAgent("white")

    episode = 0
    b_wins = w_wins = draws = 0
    latest = find_latest_snapshot()
    if latest > 0:
        snap = snapshot_dir(latest)
        print(f"[恢复] 发现最新快照：{snap}")
        meta = load_meta(os.path.join(snap, "meta.json"))
        if meta:
            episode, b_wins, w_wins, draws = meta["episode"], meta["b_wins"], meta["w_wins"], meta["draws"]
            print(f"[meta] episode={episode}, 黑胜={b_wins}, 白胜={w_wins}, 平={draws}")
        agent_black.load_json(os.path.join(snap, "black.json"))
        agent_white.load_json(os.path.join(snap, "white.json"))
    else:
        print("[提示] 未发现历史快照，从零开始训练。")

    print(f"\n开始训练... 当前 episode={episode}\n")

    try:
        while True:
            winner, moves, avg_loss = train_step(agent_black, agent_white, game)
            episode += 1
            if winner == 1: b_wins += 1
            elif winner == 2: w_wins += 1
            else: draws += 1

            if episode % TARGET_UPDATE_INTERVAL == 0:
                agent_black.update_target()
                agent_white.update_target()

            w_str = "黑胜" if winner == 1 else "白胜" if winner == 2 else "平局"
            print(f"[{episode:>6}] {w_str} | 步数={moves:>3} | loss={avg_loss:.4f} | ε={agent_black.epsilon:.3f}")

            if episode % AUTO_SAVE_INTERVAL == 0:
                snap = snapshot_dir(episode)
                agent_black.save_json(os.path.join(snap, "black.json"))
                agent_white.save_json(os.path.join(snap, "white.json"))
                save_meta(os.path.join(snap, "meta.json"), episode, b_wins, w_wins, draws)
                print(f"==== 自动保存 @ {episode} 局 ====\n")

    except KeyboardInterrupt:
        save_ep = ((episode // AUTO_SAVE_INTERVAL) + 1) * AUTO_SAVE_INTERVAL
        snap = snapshot_dir(save_ep)
        print(f"\n手动中断，保存到 {snap} ...")
        agent_black.save_json(os.path.join(snap, "black.json"))
        agent_white.save_json(os.path.join(snap, "white.json"))
        save_meta(os.path.join(snap, "meta.json"), episode, b_wins, w_wins, draws)
        print("已保存，退出。")

if __name__ == "__main__":
    main()