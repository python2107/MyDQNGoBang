# ============================================================
#  五子棋 DQN 训练 (ESP32-S3 版 · 13 路)
#  ulab 矩阵加速 · JSON 存档格式 v1 · 与 PC 版 train.py 互通
#
#  存档路径：/qgnn13/black.json  /qgnn13/white.json  /qgnn13/meta.json
#  与 PC 版的对应关系：
#     PC 端 SAVE_DIR = "qgnn13"     (Windows/Linux 相对路径)
#     ESP32 SAVE_DIR = "/qgnn13"    (MicroPython 绝对路径)
#   用 Thonny 直接拖放三个文件即可互传。
# ============================================================
import gc
import os
import json
import time
import random
import ulab.numpy as np

# ==================== 全局配置 ====================
BOARD_SIZE   = 13
HIDDEN       = 256
ACTION_SIZE  = BOARD_SIZE * BOARD_SIZE       # 169
STATE_SIZE   = ACTION_SIZE

FMT_VERSION  = 1
SAVE_DIR     = "/qgnn13"

# 与 PC 版对齐的超参数（epsilon 曲线保持一致，便于互传后继续训练）
LR              = 0.001
GAMMA           = 0.99
EPSILON_START   = 1.0
EPSILON_MIN     = 0.01
EPSILON_DECAY   = 0.995
TARGET_UPDATE_INTERVAL = 10

# ESP32 专用：因为内存和速度限制，batch 和 replay 步数需要单独设
BATCH_SIZE      = 4          # PC 版是 64；ESP32 上 4 足够且不 OOM
MEMORY_CAPACITY = 300        # PC 版是 20000
REPLAY_STEPS    = 4          # 每次 learn 做几次梯度更新

AUTO_SAVE_INTERVAL = 100     # 每 N 局保存一次（写 JSON 慢，别太频繁）
LOG_INTERVAL       = 5
GC_INTERVAL        = 20
MIN_MEM_TO_LEARN   = BATCH_SIZE * 4

# 九宫格奖励
REGION_REWARD = [
    [0.05, 0.02, 0.05],
    [0.02, 0.10, 0.02],
    [0.05, 0.02, 0.05],
]
SHAPE_REWARD = {
    'live_four':   0.5,
    'rush_four':   0.2,
    'live_three':  0.1,
    'sleep_three': 0.02,
    'live_two':    0.01,
}
THREAT_PENALTY = {
    'live_four':  -0.5,
    'rush_four':  -0.2,
    'live_three': -0.1,
}

try:
    os.mkdir(SAVE_DIR)
except OSError:
    pass
# ==================================================


# -------------------- 数值辅助 (尽量走 ulab) --------------------
def _randn(rows, cols, scale):
    """生成 (rows, cols) 高斯矩阵。初始化慢一点没关系。"""
    data = [random.gauss(0.0, scale) for _ in range(rows * cols)]
    return np.array(data).reshape((rows, cols))


def _relu(M):
    try:
        return np.maximum(M, 0.0)
    except AttributeError:
        return (M + abs(M)) * 0.5


def _relu_back(Z, dZ):
    """dZ * (Z > 0)，全部数组运算。"""
    try:
        return np.where(Z > 0, dZ, 0.0)
    except AttributeError:
        mask = (Z > 0)
        return dZ * mask


def _row_broadcast(b, B):
    """(H,) 向量扩展成 (B, H) 每行相同。"""
    try:
        return np.outer(np.ones(B), b)
    except AttributeError:
        H = len(b)
        out = np.zeros((B, H))
        for i in range(B):
            for j in range(H):
                out[i, j] = b[j]
        return out


def _sum_axis0(M):
    """按列求和 -> (H,)。"""
    try:
        return np.sum(M, axis=0)
    except (AttributeError, TypeError):
        B, H = M.shape
        out = np.zeros(H)
        for i in range(B):
            out = out + M[i]
        return out


def _flatten_2d(nested):
    """快速扁平化 2D 嵌套列表（比通用递归快 5~10 倍）。"""
    out = []
    for row in nested:
        out.extend(row)
    return out


# -------------------- 游戏环境 --------------------
class GoBangGame:
    def __init__(self):
        self.size = BOARD_SIZE
        self.board = None
        self.current_player = None
        self.winner = None
        self.reset()

    def reset(self):
        S = self.size
        self.board = [[0] * S for _ in range(S)]
        self.current_player = 1
        self.winner = None

    def get_state(self):
        """从当前走子方视角返回 (S*S,) float32: 自己+1, 对手-1, 空0。"""
        S = self.size
        p = self.current_player
        opp = 3 - p
        s = np.zeros(S * S)
        k = 0
        b = self.board
        for i in range(S):
            row = b[i]
            for j in range(S):
                v = row[j]
                if v == p:
                    s[k] = 1.0
                elif v == opp:
                    s[k] = -1.0
                k += 1
        return s

    def get_valid_actions_idx(self):
        S = self.size
        b = self.board
        out = []
        for i in range(S):
            row = b[i]
            base = i * S
            for j in range(S):
                if row[j] == 0:
                    out.append(base + j)
        return out

    def _shape_reward(self, i, j, player):
        S = self.size
        b = self.board
        reward = 0.0
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            length = 1
            open_ends = 0
            x, y = i + dx, j + dy
            while 0 <= x < S and 0 <= y < S and b[x][y] == player:
                length += 1; x += dx; y += dy
            if 0 <= x < S and 0 <= y < S and b[x][y] == 0:
                open_ends += 1
            x, y = i - dx, j - dy
            while 0 <= x < S and 0 <= y < S and b[x][y] == player:
                length += 1; x -= dx; y -= dy
            if 0 <= x < S and 0 <= y < S and b[x][y] == 0:
                open_ends += 1

            if length >= 5:
                continue
            elif length == 4:
                if open_ends == 2:   reward += SHAPE_REWARD['live_four']
                elif open_ends == 1: reward += SHAPE_REWARD['rush_four']
            elif length == 3:
                if open_ends == 2:   reward += SHAPE_REWARD['live_three']
                elif open_ends == 1: reward += SHAPE_REWARD['sleep_three']
            elif length == 2 and open_ends == 2:
                reward += SHAPE_REWARD['live_two']
        return reward

    def _threat_penalty(self, opponent):
        S = self.size
        b = self.board
        penalty = 0.0
        for i in range(S):
            for j in range(S):
                if b[i][j] != opponent:
                    continue
                for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    x, y = i - dx, j - dy
                    if 0 <= x < S and 0 <= y < S and b[x][y] == opponent:
                        continue
                    length = 0
                    x, y = i, j
                    while 0 <= x < S and 0 <= y < S and b[x][y] == opponent:
                        length += 1; x += dx; y += dy
                    open_ends = 0
                    x1, y1 = i - dx, j - dy
                    if 0 <= x1 < S and 0 <= y1 < S and b[x1][y1] == 0:
                        open_ends += 1
                    x2 = i + (length - 1) * dx + dx
                    y2 = j + (length - 1) * dy + dy
                    if 0 <= x2 < S and 0 <= y2 < S and b[x2][y2] == 0:
                        open_ends += 1
                    if length >= 5:
                        continue
                    if length == 4:
                        if open_ends == 2:   penalty += THREAT_PENALTY['live_four']
                        elif open_ends == 1: penalty += THREAT_PENALTY['rush_four']
                    elif length == 3 and open_ends == 2:
                        penalty += THREAT_PENALTY['live_three']
        return penalty

    def make_move(self, action_idx):
        S = self.size
        i = action_idx // S
        j = action_idx % S
        if self.board[i][j] != 0:
            return 0.0, False

        self.board[i][j] = self.current_player

        region_i = min(i // (S // 3), 2)
        region_j = min(j // (S // 3), 2)
        region_reward = REGION_REWARD[region_i][region_j]
        shape_reward = self._shape_reward(i, j, self.current_player)
        opponent = 3 - self.current_player
        threat_penalty = self._threat_penalty(opponent)
        total_reward = region_reward + shape_reward + threat_penalty

        done = False
        if self._check_winner(i, j):
            self.winner = self.current_player
            done = True
        elif not self.get_valid_actions_idx():
            self.winner = 0
            done = True
        else:
            self.current_player = opponent
        return total_reward, done

    def _check_winner(self, i, j):
        S = self.size
        b = self.board
        p = b[i][j]
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            cnt = 1
            for step in (1, -1):
                x, y = i + dx * step, j + dy * step
                while 0 <= x < S and 0 <= y < S and b[x][y] == p:
                    cnt += 1; x += dx * step; y += dy * step
            if cnt >= 5:
                return True
        return False


# -------------------- Q 网络 --------------------
class QNet:
    """
    内部权重形状 (in, out)，便于 np.dot(X, W) 直接算:
        W1: (STATE_SIZE, HIDDEN) = (169, 256)
        W2: (HIDDEN, HIDDEN)     = (256, 256)
        W3: (HIDDEN, ACTION_SIZE)= (256, 169)
    JSON 里存 PyTorch 原生 (out, in):
        fc1.weight: (HIDDEN, STATE_SIZE)
        fc2.weight: (HIDDEN, HIDDEN)
        fc3.weight: (ACTION_SIZE, HIDDEN)
    """

    def __init__(self, n_in, n_out, h):
        self.n_in = n_in
        self.n_out = n_out
        self.h = h
        s1 = (2.0 / n_in) ** 0.5
        s2 = (2.0 / h) ** 0.5
        self.W1 = _randn(n_in, h, s1); self.b1 = np.zeros(h)
        self.W2 = _randn(h, h, s2);    self.b2 = np.zeros(h)
        self.W3 = _randn(h, n_out, s2); self.b3 = np.zeros(n_out)

    def forward(self, X):
        B = X.shape[0]
        Z1 = np.dot(X, self.W1) + _row_broadcast(self.b1, B)
        A1 = _relu(Z1)
        Z2 = np.dot(A1, self.W2) + _row_broadcast(self.b2, B)
        A2 = _relu(Z2)
        Q  = np.dot(A2, self.W3) + _row_broadcast(self.b3, B)
        return Q, (X, Z1, A1, Z2, A2)

    def backward(self, cache, dQ, lr):
        X, Z1, A1, Z2, A2 = cache
        dW3 = np.dot(A2.T, dQ);  db3 = _sum_axis0(dQ)
        dA2 = np.dot(dQ, self.W3.T)
        dZ2 = _relu_back(Z2, dA2)
        dW2 = np.dot(A1.T, dZ2); db2 = _sum_axis0(dZ2)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = _relu_back(Z1, dA1)
        dW1 = np.dot(X.T, dZ1);  db1 = _sum_axis0(dZ1)

        self.W3 = self.W3 - lr * dW3; self.b3 = self.b3 - lr * db3
        self.W2 = self.W2 - lr * dW2; self.b2 = self.b2 - lr * db2
        self.W1 = self.W1 - lr * dW1; self.b1 = self.b1 - lr * db1

    def copy_from(self, other):
        self.W1 = np.array(other.W1); self.b1 = np.array(other.b1)
        self.W2 = np.array(other.W2); self.b2 = np.array(other.b2)
        self.W3 = np.array(other.W3); self.b3 = np.array(other.b3)

    # ---- JSON 序列化：与 PC 版完全一致 ----
    def to_json_dict(self):
        """内部 (in,out) 转置为 PyTorch 原生 (out,in) 的嵌套列表。"""
        W1T = np.array(self.W1.T)   # (h, n_in)
        W2T = np.array(self.W2.T)   # (h, h)
        W3T = np.array(self.W3.T)   # (n_out, h)
        return {
            "fc1.weight": [W1T[i].tolist() for i in range(self.h)],
            "fc1.bias":   self.b1.tolist(),
            "fc2.weight": [W2T[i].tolist() for i in range(self.h)],
            "fc2.bias":   self.b2.tolist(),
            "fc3.weight": [W3T[i].tolist() for i in range(self.n_out)],
            "fc3.bias":   self.b3.tolist(),
        }

    def from_json_dict(self, d):
        """从 (out,in) 嵌套列表加载，转置回内部 (in,out)。"""
        n_in, h, n_out = self.n_in, self.h, self.n_out

        W1T = np.array(_flatten_2d(d["fc1.weight"])).reshape((h, n_in))
        self.W1 = np.array(W1T.T)   # (n_in, h) 连续
        self.b1 = np.array(d["fc1.bias"])

        W2T = np.array(_flatten_2d(d["fc2.weight"])).reshape((h, h))
        self.W2 = np.array(W2T.T)
        self.b2 = np.array(d["fc2.bias"])

        W3T = np.array(_flatten_2d(d["fc3.weight"])).reshape((n_out, h))
        self.W3 = np.array(W3T.T)
        self.b3 = np.array(d["fc3.bias"])


# -------------------- DQN Agent --------------------
class DQNAgent:
    def __init__(self, name="agent", save_path=None):
        self.name = name
        self.save_path = save_path
        self.model = QNet(STATE_SIZE, ACTION_SIZE, HIDDEN)
        self.target = QNet(STATE_SIZE, ACTION_SIZE, HIDDEN)
        self.target.copy_from(self.model)
        self.memory = []
        self.mem_idx = 0
        self.epsilon = EPSILON_START
        self.updates = 0
        self.last_loss = 0.0

    # ---- 决策 ----
    def act(self, state, valid_actions):
        if random.random() < self.epsilon:
            return valid_actions[random.randrange(len(valid_actions))]
        X = state.reshape((1, STATE_SIZE))
        Q, _ = self.model.forward(X)
        best_a = valid_actions[0]
        best_v = Q[0, best_a]
        for a in valid_actions:
            v = Q[0, a]
            if v > best_v:
                best_v = v
                best_a = a
        return best_a

    def remember(self, s, a, r, ns, d):
        if len(self.memory) < MEMORY_CAPACITY:
            self.memory.append((s, a, r, ns, d))
        else:
            self.memory[self.mem_idx] = (s, a, r, ns, d)
            self.mem_idx = (self.mem_idx + 1) % MEMORY_CAPACITY

    # ---- 梯度更新 ----
    def learn(self):
        if len(self.memory) < MIN_MEM_TO_LEARN:
            return None

        total_loss = 0.0
        for _ in range(REPLAY_STEPS):
            batch = random.sample(self.memory, BATCH_SIZE)
            X  = np.zeros((BATCH_SIZE, STATE_SIZE))
            NS = np.zeros((BATCH_SIZE, STATE_SIZE))
            actions = [0] * BATCH_SIZE
            rewards = [0.0] * BATCH_SIZE
            dones   = [False] * BATCH_SIZE

            for i in range(BATCH_SIZE):
                s, a, r, ns, d = batch[i]
                for j in range(STATE_SIZE):
                    X[i, j]  = s[j]
                    NS[i, j] = ns[j]
                actions[i] = a
                rewards[i] = r
                dones[i]   = d

            Q, cache = self.model.forward(X)
            TQ, _    = self.target.forward(NS)

            dQ = np.zeros((BATCH_SIZE, ACTION_SIZE))
            for i in range(BATCH_SIZE):
                if dones[i]:
                    target = rewards[i]
                else:
                    best = -1e30
                    ns_i = NS[i]
                    for a in range(ACTION_SIZE):
                        if ns_i[a] == 0.0:
                            v = TQ[i, a]
                            if v > best:
                                best = v
                    if best < -1e29:
                        best = 0.0
                    target = rewards[i] + GAMMA * best
                td = Q[i, actions[i]] - target
                dQ[i, actions[i]] = td
                total_loss += td * td

            self.model.backward(cache, dQ, LR)

        # ε 衰减（每次 learn 只衰减一次，与 PC 版每局一次 replay 对齐）
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
        self.updates += 1

        self.last_loss = total_loss / (REPLAY_STEPS * BATCH_SIZE)
        return self.last_loss

    def update_target(self):
        self.target.copy_from(self.model)

    # ---- JSON 保存（覆盖写，只留最新） ----
    def save_json(self, path=None):
        if path is None:
            path = self.save_path
        payload = {
            "fmt":     FMT_VERSION,
            "board":   BOARD_SIZE,
            "hidden":  HIDDEN,
            "epsilon": float(self.epsilon),
            "updates": int(self.updates),
            "model":   self.model.to_json_dict(),
        }
        with open(path, "w") as f:
            json.dump(payload, f)
        gc.collect()

    # ---- JSON 加载 ----
    def load_json(self, path=None):
        if path is None:
            path = self.save_path
        if not os.path.exists(path):
            print("  [加载] %s 不存在，跳过。" % path)
            return False
        try:
            with open(path, "r") as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            print("  [加载] %s 解析失败：%s" % (path, e))
            return False

        if d.get("fmt") != FMT_VERSION:
            print("  [加载] fmt 不匹配：%s != %s" % (d.get("fmt"), FMT_VERSION))
            return False
        if d.get("board") != BOARD_SIZE:
            print("  [加载] board 不匹配：%s != %s" % (d.get("board"), BOARD_SIZE))
            return False
        if d.get("hidden") != HIDDEN:
            print("  [加载] hidden 不匹配：%s != %s" % (d.get("hidden"), HIDDEN))
            return False

        try:
            self.model.from_json_dict(d["model"])
            self.target.copy_from(self.model)
        except Exception as e:
            print("  [加载] 权重载入失败：%s" % e)
            return False

        self.epsilon = float(d.get("epsilon", EPSILON_START))
        self.updates = int(d.get("updates", 0))
        print("  [加载] %s 成功 (ε=%.4f, updates=%d)"
              % (path, self.epsilon, self.updates))
        return True


# -------------------- meta.json --------------------
def save_meta(path, episode, b_wins, w_wins, draws):
    payload = {
        "fmt":     FMT_VERSION,
        "board":   BOARD_SIZE,
        "hidden":  HIDDEN,
        "episode": int(episode),
        "b_wins":  int(b_wins),
        "w_wins":  int(w_wins),
        "draws":   int(draws),
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def load_meta(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if (d.get("fmt") != FMT_VERSION
            or d.get("board") != BOARD_SIZE
            or d.get("hidden") != HIDDEN):
        print("  [meta] 校验失败，忽略旧进度。")
        return None
    return d


# -------------------- 训练一轮 --------------------
def train_step(agent_black, agent_white, game):
    """跑一整局，收集经验并做一次梯度更新。返回 (winner, moves, loss)。"""
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
        history.append((player, state, action, reward, next_state, done))
        moves += 1

    winner = game.winner
    # 经验回填
    for p, s, a, r, ns, d in history:
        total = r + (1.0 if (d and winner == p) else 0.0)
        if p == 1:
            agent_black.remember(s, a, total, ns, d)
        else:
            agent_white.remember(s, a, total, ns, d)

    lb = agent_black.learn()
    lw = agent_white.learn()
    losses = [l for l in (lb, lw) if l is not None]
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return winner, moves, avg_loss


# -------------------- 主循环 --------------------
def main():
    game = GoBangGame()
    black_path = SAVE_DIR + "/black.json"
    white_path = SAVE_DIR + "/white.json"
    meta_path  = SAVE_DIR + "/meta.json"

    agent_black = DQNAgent("black", black_path)
    agent_white = DQNAgent("white", white_path)

    # ---- 恢复进度 ----
    episode = 0
    b_wins = w_wins = draws = 0

    meta = load_meta(meta_path)
    if meta is not None:
        episode = int(meta["episode"])
        b_wins  = int(meta["b_wins"])
        w_wins  = int(meta["w_wins"])
        draws   = int(meta["draws"])
        print("[meta] 恢复进度：episode=%d, 黑胜=%d, 白胜=%d, 平=%d"
              % (episode, b_wins, w_wins, draws))

    ok_b = agent_black.load_json(black_path)
    ok_w = agent_white.load_json(white_path)
    if not (ok_b and ok_w):
        print("[提示] 权重未加载或校验失败，使用随机初始化。")

    print("=" * 62)
    print("五子棋 DQN 训练 (ESP32-S3 · 13x13 · JSON v%d)" % FMT_VERSION)
    print("棋盘 %dx%d | HIDDEN %d | 动作 %d | batch %d | mem %d"
          % (BOARD_SIZE, BOARD_SIZE, HIDDEN, ACTION_SIZE,
             BATCH_SIZE, MEMORY_CAPACITY))
    print("存档: %s" % SAVE_DIR)
    print("起始 episode=%d | Ctrl+C 中断并保存" % episode)
    print("=" * 62)

    t_start = time.time()
    start_ep = episode
    total_eps = max(episode - start_ep, 1)

    try:
        while True:
            winner, moves, avg_loss = train_step(agent_black, agent_white, game)
            episode += 1

            if winner == 1:
                b_wins += 1
            elif winner == 2:
                w_wins += 1
            else:
                draws += 1

            if episode % TARGET_UPDATE_INTERVAL == 0:
                agent_black.update_target()
                agent_white.update_target()

            if episode % LOG_INTERVAL == 0:
                elapsed = time.time() - t_start
                n_sess = episode - start_ep
                w_str = "黑胜" if winner == 1 else ("白胜" if winner == 2 else "平局")
                tot = max(b_wins + w_wins + draws, 1)
                print("[%6d] %s | 步数=%3d | loss=%.4f | "
                      "黑 %d(%.1f%%) 白 %d(%.1f%%) 平 %d | "
                      "ε=%.4f/%.4f | mem %d | %.1fs/ep | free %d"
                      % (episode, w_str, moves, avg_loss,
                         b_wins, 100.0 * b_wins / tot,
                         w_wins, 100.0 * w_wins / tot, draws,
                         agent_black.epsilon, agent_white.epsilon,
                         len(agent_black.memory),
                         elapsed / max(n_sess, 1),
                         gc.mem_free()))

            if episode % AUTO_SAVE_INTERVAL == 0:
                print("==== 自动保存 @ %d 局 ====" % episode)
                agent_black.save_json(black_path)
                agent_white.save_json(white_path)
                save_meta(meta_path, episode, b_wins, w_wins, draws)
                print("==== 保存完成 ====")

            if episode % GC_INTERVAL == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("\n手动中断，保存当前进度 @ %d 局 ..." % episode)
    except Exception as e:
        print("\n训练异常中断：%s" % repr(e))
        try:
            import sys
            sys.print_exception(e)
        except Exception:
            pass
    finally:
        agent_black.save_json(black_path)
        agent_white.save_json(white_path)
        save_meta(meta_path, episode, b_wins, w_wins, draws)
        print("已保存到 %s (第 %d 局)" % (SAVE_DIR, episode))


if __name__ == "__main__":
    main()
