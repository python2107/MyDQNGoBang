import tkinter as tk
from tkinter import messagebox
import numpy as np
import random
import json
import os
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ===================== 参数配置 =====================
BOARD_SIZE = 13
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
STATE_SIZE = ACTION_SIZE

# 界面刷新间隔（每 N 步刷新一次棋盘）
REFRESH_INTERVAL = 5
# 训练频率（每 N 步执行一次 replay）
TRAIN_INTERVAL = 2
# 网络隐藏层大小
HIDDEN_SIZE = 128

# 九宫格奖励
REGION_REWARD = [
    [0.05,  0.02,  0.05],
    [0.02,  0.10,  0.02],
    [0.05,  0.02,  0.05]
]

# 棋型奖励（正数）
SHAPE_REWARD = {
    'live_four': 0.5,
    'rush_four': 0.2,
    'live_three': 0.1,
    'sleep_three': 0.02,
    'live_two': 0.01,
}

# 对手威胁惩罚（负数）
THREAT_PENALTY = {
    'live_four': -0.5,
    'rush_four': -0.2,
    'live_three': -0.1,
}

# DQN 超参数
LR = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_CAPACITY = 10000
TARGET_UPDATE_INTERVAL = 10

# 自动保存间隔（局数）
AUTO_SAVE_INTERVAL = 500

# 权重保存根目录
WEIGHT_ROOT = "json/bogang"
# ====================================================

os.makedirs(WEIGHT_ROOT, exist_ok=True)

# --------------------------- 游戏环境 ---------------------------
class GoBangGame:
    def __init__(self):
        self.size = BOARD_SIZE
        self.board = None
        self.current_player = None
        self.winner = None
        self.reset()

    def reset(self):
        self.board = [[0]*self.size for _ in range(self.size)]
        self.current_player = 1
        self.winner = None
        return self.get_state()

    def get_state(self):
        state = np.zeros((self.size, self.size), dtype=np.float32)
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 1:
                    state[i][j] = 1
                elif self.board[i][j] == 2:
                    state[i][j] = -1
        return state

    def get_valid_actions_idx(self):
        idx = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    idx.append(i*self.size + j)
        return idx

    def get_shape_reward(self, i, j, player):
        reward = 0.0
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for dx, dy in directions:
            length = 1
            open_ends = 0
            x, y = i+dx, j+dy
            while 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == player:
                length += 1
                x += dx
                y += dy
            if 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == 0:
                open_ends += 1
            x, y = i-dx, j-dy
            while 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == player:
                length += 1
                x -= dx
                y -= dy
            if 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == 0:
                open_ends += 1

            if length >= 5:
                continue
            elif length == 4:
                if open_ends == 2:
                    reward += SHAPE_REWARD.get('live_four', 0)
                elif open_ends == 1:
                    reward += SHAPE_REWARD.get('rush_four', 0)
            elif length == 3:
                if open_ends == 2:
                    reward += SHAPE_REWARD.get('live_three', 0)
                elif open_ends == 1:
                    reward += SHAPE_REWARD.get('sleep_three', 0)
            elif length == 2 and open_ends == 2:
                reward += SHAPE_REWARD.get('live_two', 0)
        return reward

    def get_opponent_threat_penalty(self, opponent):
        """全局威胁检测：遍历所有对手棋子，统计活三、冲四、活四"""
        penalty = 0.0
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != opponent:
                    continue
                for dx, dy in directions:
                    x, y = i-dx, j-dy
                    if 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == opponent:
                        continue
                    length = 0
                    x, y = i, j
                    while 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == opponent:
                        length += 1
                        x += dx
                        y += dy
                    open_ends = 0
                    x1, y1 = i-dx, j-dy
                    if 0 <= x1 < self.size and 0 <= y1 < self.size and self.board[x1][y1] == 0:
                        open_ends += 1
                    x2, y2 = i + (length-1)*dx, j + (length-1)*dy
                    x2 += dx
                    y2 += dy
                    if 0 <= x2 < self.size and 0 <= y2 < self.size and self.board[x2][y2] == 0:
                        open_ends += 1

                    if length >= 5:
                        continue
                    if length == 4:
                        if open_ends == 2:
                            penalty += THREAT_PENALTY.get('live_four', 0)
                        elif open_ends == 1:
                            penalty += THREAT_PENALTY.get('rush_four', 0)
                    elif length == 3 and open_ends == 2:
                        penalty += THREAT_PENALTY.get('live_three', 0)
        return penalty

    def make_move(self, action_idx):
        i = action_idx // self.size
        j = action_idx % self.size
        if self.board[i][j] != 0:
            return 0.0, False

        self.board[i][j] = self.current_player

        region_i = min(i // (self.size//3), 2)
        region_j = min(j // (self.size//3), 2)
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
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        for dx, dy in directions:
            count = 1
            for step in (1, -1):
                x, y = i + dx*step, j + dy*step
                while 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == player:
                    count += 1
                    x += dx*step
                    y += dy*step
            if count >= 5:
                return True
        return False

# --------------------------- DQN 网络 ---------------------------
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, ACTION_SIZE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

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
        state_t = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model(state_t).cpu().numpy().flatten()
        valid_set = set(valid_actions)
        for idx in range(ACTION_SIZE):
            if idx not in valid_set:
                q[idx] = -np.inf
        return int(np.argmax(q))

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target(next_states).max(1)[0]
        target_q = rewards + (1 - dones.float()) * GAMMA * next_q
        loss = F.mse_loss(current_q, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def save(self, filepath):
        data = {
            'model': {k: v.cpu().numpy().tolist() for k, v in self.model.state_dict().items()},
            'epsilon': self.epsilon
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath):
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            state_dict = {k: torch.tensor(v) for k, v in data['model'].items()}
            model_state = self.model.state_dict()
            for k in state_dict.keys():
                if k in model_state:
                    if state_dict[k].shape != model_state[k].shape:
                        print(f"形状不匹配，跳过加载 {filepath}")
                        return False
            self.model.load_state_dict(state_dict)
            self.target.load_state_dict(state_dict)
            self.epsilon = data.get('epsilon', EPSILON_START)
            return True
        except Exception as e:
            print(f"加载失败: {e}")
            return False

# --------------------------- Tkinter 界面 ---------------------------
class GoBangGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DQN 五子棋 (13路, 无限训练)")

        self.canvas = tk.Canvas(root, width=600, height=600, bg='#DEB887')
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        frame = tk.Frame(root)
        frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        # 控制按钮
        self.btn_start = tk.Button(frame, text="开始训练", command=self.start_training)
        self.btn_start.pack(pady=5)

        self.btn_pause = tk.Button(frame, text="暂停", command=self.pause_training, state=tk.DISABLED)
        self.btn_pause.pack(pady=5)

        self.btn_stop = tk.Button(frame, text="停止", command=self.stop_training, state=tk.DISABLED)
        self.btn_stop.pack(pady=5)

        # 后台极速模式
        self.bg_mode = tk.BooleanVar(value=False)
        tk.Checkbutton(frame, text="后台极速模式 (不刷新棋盘)",
                       variable=self.bg_mode).pack(pady=5)

        # 加载权重区域
        load_frame = tk.Frame(frame)
        load_frame.pack(pady=10)
        tk.Label(load_frame, text="加载轮数:").pack(side=tk.LEFT)
        self.load_entry = tk.Entry(load_frame, width=10)
        self.load_entry.pack(side=tk.LEFT, padx=5)
        self.btn_load = tk.Button(load_frame, text="加载", command=self.load_weights_by_round)
        self.btn_load.pack(side=tk.LEFT)

        self.btn_save = tk.Button(frame, text="保存权重", command=self.save_weights)
        self.btn_save.pack(pady=5)

        # 信息显示
        self.info = tk.Label(frame, text="状态: 就绪", font=('Arial', 12))
        self.info.pack(pady=10)

        self.ep_label = tk.Label(frame, text="当前局数: 0")
        self.ep_label.pack()

        self.loaded_label = tk.Label(frame, text="已加载轮数: 0")
        self.loaded_label.pack()

        self.move_label = tk.Label(frame, text="步数: 0")
        self.move_label.pack()

        self.cell_size = 600 // BOARD_SIZE
        self.piece_radius = self.cell_size//2 - 3

        self.game = GoBangGame()
        self.black = DQNAgent()
        self.white = DQNAgent()

        self.training = False
        self.paused = False
        self.episode = 0
        self.move_cnt = 0
        self.history = []
        self.after_id = None
        self.step_counter = 0
        self.loaded_round = 0

        self.draw_board()

    def draw_board(self, state=None):
        self.canvas.delete("all")
        size = BOARD_SIZE
        for i in range(size):
            x = i*self.cell_size + self.cell_size//2
            y = i*self.cell_size + self.cell_size//2
            self.canvas.create_line(x, self.cell_size//2, x, 600-self.cell_size//2, fill='black')
            self.canvas.create_line(self.cell_size//2, y, 600-self.cell_size//2, y, fill='black')
        stars = [(3,3),(3,9),(9,3),(9,9),(6,6)] if size==13 else []
        for i,j in stars:
            x = i*self.cell_size + self.cell_size//2
            y = j*self.cell_size + self.cell_size//2
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill='black')

        if state is None:
            state = self.game.get_state()
        for i in range(size):
            for j in range(size):
                val = state[i][j]
                if val != 0:
                    x = j*self.cell_size + self.cell_size//2
                    y = i*self.cell_size + self.cell_size//2
                    color = 'black' if val==1 else 'white'
                    self.canvas.create_oval(x-self.piece_radius, y-self.piece_radius,
                                            x+self.piece_radius, y+self.piece_radius,
                                            fill=color, outline='gray')
        self.canvas.update()

    def start_training(self):
        if self.training:
            return
        self.training = True
        self.paused = False
        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL, text="暂停")
        self.btn_stop.config(state=tk.NORMAL)
        self.info.config(text="状态: 训练中...")
        self.game.reset()
        self.history = []
        self.move_cnt = 0
        self.step_counter = 0
        self.step_training()

    def pause_training(self):
        if self.training:
            self.paused = not self.paused
            self.btn_pause.config(text="继续" if self.paused else "暂停")
            if not self.paused:
                self.step_training()

    def stop_training(self):
        self.training = False
        self.paused = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.btn_start.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED, text="暂停")
        self.btn_stop.config(state=tk.DISABLED)
        self.info.config(text="状态: 已停止")

    def step_training(self):
        if not self.training or self.paused:
            return

        if self.game.winner is not None:
            self.end_episode()
            self.game.reset()
            self.history = []
            self.move_cnt = 0
            self.episode += 1
            self.ep_label.config(text=f"当前局数: {self.episode}")

            if self.episode % TARGET_UPDATE_INTERVAL == 0:
                self.black.update_target()
                self.white.update_target()

            if self.episode % AUTO_SAVE_INTERVAL == 0:
                self.auto_save(self.episode)

            self.step_training()
            return

        player = self.game.current_player
        agent = self.black if player==1 else self.white

        state = self.game.get_state()
        valid = self.game.get_valid_actions_idx()
        if not valid:
            self.game.winner = 0
            self.step_training()
            return

        action = agent.act(state, valid)
        reward_im, done = self.game.make_move(action)
        next_state = self.game.get_state()
        self.move_cnt += 1
        self.move_label.config(text=f"步数: {self.move_cnt}")
        self.step_counter += 1

        self.history.append((player, state.copy(), action, reward_im, next_state.copy(), done))

        # 刷新界面（根据后台模式）
        if self.step_counter % REFRESH_INTERVAL == 0 or done:
            if not self.bg_mode.get():
                self.draw_board(next_state)
                self.root.update_idletasks()
            else:
                # 只更新文本标签
                self.ep_label.config(text=f"当前局数: {self.episode}")
                self.move_label.config(text=f"步数: {self.move_cnt}")
                self.root.update_idletasks()

        if done:
            winner = self.game.winner
            for p, s, a, r_im, ns, d in self.history:
                total = r_im + (1.0 if d and winner==p else 0.0)
                if p == 1:
                    self.black.remember(s, a, total, ns, d)
                else:
                    self.white.remember(s, a, total, ns, d)
            self.black.replay()
            self.white.replay()
            self.after_id = self.root.after(0, self.step_training)
        else:
            if self.step_counter % TRAIN_INTERVAL == 0:
                self.black.replay()
                self.white.replay()
            self.after_id = self.root.after(0, self.step_training)

    def end_episode(self):
        w = self.game.winner
        msg = "黑胜" if w==1 else "白胜" if w==2 else "平局"
        self.info.config(text=f"状态: {msg}")

    def auto_save(self, round_num):
        save_dir = os.path.join(WEIGHT_ROOT, str(round_num))
        os.makedirs(save_dir, exist_ok=True)
        black_path = os.path.join(save_dir, "black.json")
        white_path = os.path.join(save_dir, "white.json")
        self.black.save(black_path)
        self.white.save(white_path)
        self.loaded_round = round_num
        self.loaded_label.config(text=f"已加载轮数: {round_num}")
        print(f"自动保存: {save_dir}")

    def save_weights(self):
        if self.episode == 0:
            messagebox.showwarning("警告", "当前没有完成任何对局，无法保存。")
            return
        save_dir = os.path.join(WEIGHT_ROOT, str(self.episode))
        os.makedirs(save_dir, exist_ok=True)
        black_path = os.path.join(save_dir, "black.json")
        white_path = os.path.join(save_dir, "white.json")
        self.black.save(black_path)
        self.white.save(white_path)
        self.loaded_round = self.episode
        self.loaded_label.config(text=f"已加载轮数: {self.episode}")
        messagebox.showinfo("保存", f"权重已保存至 {save_dir}")

    def load_weights_by_round(self):
        round_str = self.load_entry.get().strip()
        if not round_str.isdigit():
            messagebox.showerror("错误", "请输入有效的数字")
            return
        round_num = int(round_str)
        load_dir = os.path.join(WEIGHT_ROOT, str(round_num))
        black_path = os.path.join(load_dir, "black.json")
        white_path = os.path.join(load_dir, "white.json")
        if not os.path.exists(black_path) or not os.path.exists(white_path):
            messagebox.showerror("错误", f"未找到 {load_dir} 下的权重文件")
            return
        ok1 = self.black.load(black_path)
        ok2 = self.white.load(white_path)
        if ok1 and ok2:
            self.loaded_round = round_num
            self.loaded_label.config(text=f"已加载轮数: {round_num}")
            self.episode = round_num
            self.ep_label.config(text=f"当前局数: {self.episode}")
            self.game.reset()
            self.draw_board()
            messagebox.showinfo("加载", f"成功加载第 {round_num} 轮权重")
        else:
            messagebox.showerror("加载", "加载失败，可能权重格式不匹配")

if __name__ == "__main__":
    root = tk.Tk()
    app = GoBangGUI(root)
    root.mainloop()
