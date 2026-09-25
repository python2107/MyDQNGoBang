# ============================================================
#  五子棋 DQN 训练 (ESP32-S3 · 13x13 · 纯 Python 无 ulab)
#  JSON 存档 v1 · 与 PC 版 train.py 完全互通
#  存档: /qgnn13/black.json  /qgnn13/white.json  /qgnn13/meta.json
# ============================================================
import os, json, time, random, gc, math
from array import array

# ==================== 全局配置 ====================
BOARD_SIZE  = 13
HIDDEN      = 256
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
STATE_SIZE  = ACTION_SIZE
FMT_VERSION = 1
SAVE_DIR    = "/qgnn13"

LR              = 0.001
GAMMA           = 0.99
EPSILON_START   = 1.0
EPSILON_MIN     = 0.01
EPSILON_DECAY   = 0.995
TARGET_UPDATE_INTERVAL = 10

MEMORY_CAPACITY = 200
REPLAY_STEPS    = 3
MIN_MEM_TO_LEARN = 20

AUTO_SAVE_INTERVAL = 50
LOG_INTERVAL       = 1
GC_INTERVAL        = 5

REGION_REWARD = [
    [0.05, 0.02, 0.05],
    [0.02, 0.10, 0.02],
    [0.05, 0.02, 0.05],
]
SHAPE_REWARD = {
    'live_four':   0.5, 'rush_four':   0.2, 'live_three':  0.1,
    'sleep_three': 0.02, 'live_two':   0.01,
}
THREAT_PENALTY = {
    'live_four':  -0.5, 'rush_four':  -0.2, 'live_three': -0.1,
}

try:
    os.mkdir(SAVE_DIR)
except OSError:
    pass
# ==================================================


# -------------------- 工具 --------------------
def _gauss(mu, sigma):
    """Box-Muller 变换。MicroPython 没有 random.gauss。"""
    u1 = random.random()
    if u1 < 1e-10:
        u1 = 1e-10
    u2 = random.random()
    return mu + sigma * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _flatten_2d(nested):
    out = []
    for row in nested:
        out.extend(row)
    return out


# -------------------- 游戏环境 --------------------
class GoBangGame:
    def __init__(self):
        self.S = BOARD_SIZE
        self.N = self.S * self.S
        self.board = [0] * self.N
        self.current = 1
        self.winner = None

    def reset(self):
        b = self.board
        for i in range(self.N):
            b[i] = 0
        self.current = 1
        self.winner = None

    def get_state(self):
        """当前走子方视角: 自己 +1, 对手 -1, 空 0。返回 list(float) 长度 N。"""
        p = self.current
        opp = 3 - p
        b = self.board
        s = [0.0] * self.N
        for i in range(self.N):
            v = b[i]
            if v == p:
                s[i] = 1.0
            elif v == opp:
                s[i] = -1.0
        return s

    def get_valid_actions(self):
        b = self.board
        return [i for i in range(self.N) if b[i] == 0]

    def _check_winner(self, pos):
        S = self.S; b = self.board
        i = pos // S; j = pos % S
        p = b[pos]
        for dx, dy in ((1,0),(0,1),(1,1),(1,-1)):
            cnt = 1
            for step in (1, -1):
                x = i + dx * step; y = j + dy * step
                while 0 <= x < S and 0 <= y < S and b[x*S + y] == p:
                    cnt += 1
                    x += dx * step; y += dy * step
            if cnt >= 5:
                return True
        return False

    def _shape_reward(self, i, j, player):
        S = self.S; b = self.board
        reward = 0.0
        for dx, dy in ((1,0),(0,1),(1,1),(1,-1)):
            length = 1; open_ends = 0
            x = i + dx; y = j + dy
            while 0 <= x < S and 0 <= y < S and b[x*S + y] == player:
                length += 1; x += dx; y += dy
            if 0 <= x < S and 0 <= y < S and b[x*S + y] == 0:
                open_ends += 1
            x = i - dx; y = j - dy
            while 0 <= x < S and 0 <= y < S and b[x*S + y] == player:
                length += 1; x -= dx; y -= dy
            if 0 <= x < S and 0 <= y < S and b[x*S + y] == 0:
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
        S = self.S; b = self.board
        penalty = 0.0
        for i in range(S):
            for j in range(S):
                if b[i*S + j] != opponent:
                    continue
                for dx, dy in ((1,0),(0,1),(1,1),(1,-1)):
                    x = i - dx; y = j - dy
                    if 0 <= x < S and 0 <= y < S and b[x*S + y] == opponent:
                        continue
                    length = 0
                    x = i; y = j
                    while 0 <= x < S and 0 <= y < S and b[x*S + y] == opponent:
                        length += 1; x += dx; y += dy
                    open_ends = 0
                    x1 = i - dx; y1 = j - dy
                    if 0 <= x1 < S and 0 <= y1 < S and b[x1*S + y1] == 0:
                        open_ends += 1
                    x2 = i + (length - 1) * dx + dx
                    y2 = j + (length - 1) * dy + dy
                    if 0 <= x2 < S and 0 <= y2 < S and b[x2*S + y2] == 0:
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
        S = self.S
        i = action_idx // S; j = action_idx % S
        if self.board[action_idx] != 0:
            return 0.0, False
        self.board[action_idx] = self.current

        region_i = min(i // (S // 3), 2)
        region_j = min(j // (S // 3), 2)
        region_reward = REGION_REWARD[region_i][region_j]
        shape_reward  = self._shape_reward(i, j, self.current)
        opp = 3 - self.current
        threat_penalty = self._threat_penalty(opp)
        total_reward = region_reward + shape_reward + threat_penalty

        done = False
        if self._check_winner(action_idx):
            self.winner = self.current
            done = True
        elif not self.get_valid_actions():
            self.winner = 0
            done = True
        else:
            self.current = opp
        return total_reward, done


# -------------------- Q 网络 --------------------
class QNet:
    """
    array('f') 扁平存储, 与 PyTorch 原生 (out, in) 布局一致:
        W1: (HIDDEN, STATE_SIZE)
        W2: (HIDDEN, HIDDEN)
        W3: (ACTION_SIZE, HIDDEN)
    """

    def __init__(self, n_in, n_out, h):
        self.n_in = n_in
        self.n_out = n_out
        self.h = h
        s1 = (2.0 / n_in) ** 0.5
        s2 = (2.0 / h) ** 0.5

        self.W1 = array('f', [_gauss(0.0, s1) for _ in range(h * n_in)])
        self.b1 = array('f', [0.0] * h)
        self.W2 = array('f', [_gauss(0.0, s2) for _ in range(h * h)])
        self.b2 = array('f', [0.0] * h)
        self.W3 = array('f', [_gauss(0.0, s2) for _ in range(n_out * h)])
        self.b3 = array('f', [0.0] * n_out)

    def forward(self, x):
        """x: list(float) 长度 n_in。返回 (q, cache)。"""
        n_in = self.n_in; n_out = self.n_out; h = self.h
        W1 = self.W1; b1 = self.b1
        W2 = self.W2; b2 = self.b2
        W3 = self.W3; b3 = self.b3

        z1 = [0.0] * h
        for i in range(h):
            base = i * n_in
            s = b1[i]
            for k in range(n_in):
                s += W1[base + k] * x[k]
            z1[i] = s

        a1 = [0.0] * h
        for i in range(h):
            v = z1[i]
            if v > 0.0:
                a1[i] = v

        z2 = [0.0] * h
        for i in range(h):
            base = i * h
            s = b2[i]
            for k in range(h):
                s += W2[base + k] * a1[k]
            z2[i] = s

        a2 = [0.0] * h
        for i in range(h):
            v = z2[i]
            if v > 0.0:
                a2[i] = v

        q = [0.0] * n_out
        for i in range(n_out):
            base = i * h
            s = b3[i]
            for k in range(h):
                s += W3[base + k] * a2[k]
            q[i] = s

        return q, (x, z1, a1, z2, a2)

    def forward_q(self, x):
        q, _ = self.forward(x)
        return q

    def backward(self, cache, action, td, lr):
        """单样本 SGD。td = q[action] - target。"""
        x, z1, a1, z2, a2 = cache
        h = self.h; n_in = self.n_in; n_out = self.n_out

        # ---- 输出层 ----
        base_a = action * h
        dA2 = [0.0] * h
        for k in range(h):
            w = self.W3[base_a + k]
            dA2[k] = w * td
            self.W3[base_a + k] = w - lr * td * a2[k]
        self.b3[action] -= lr * td

        # ReLU 反向
        dZ2 = [0.0] * h
        for i in range(h):
            if z2[i] > 0.0:
                dZ2[i] = dA2[i]

        # ---- 第二隐层 ----
        dA1 = [0.0] * h
        for i in range(h):
            dzi = dZ2[i]
            if dzi == 0.0:
                continue
            base_i = i * h
            for k in range(h):
                w = self.W2[base_i + k]
                dA1[k] += w * dzi
                self.W2[base_i + k] = w - lr * dzi * a1[k]
            self.b2[i] -= lr * dzi

        dZ1 = [0.0] * h
        for i in range(h):
            if z1[i] > 0.0:
                dZ1[i] = dA1[i]

        # ---- 第一隐层 ----
        for i in range(h):
            dzi = dZ1[i]
            if dzi == 0.0:
                continue
            base_i = i * n_in
            for k in range(n_in):
                self.W1[base_i + k] -= lr * dzi * x[k]
            self.b1[i] -= lr * dzi

    def copy_from(self, other):
        self.W1 = array('f', other.W1)
        self.b1 = array('f', other.b1)
        self.W2 = array('f', other.W2)
        self.b2 = array('f', other.b2)
        self.W3 = array('f', other.W3)
        self.b3 = array('f', other.b3)

    # ---- JSON 序列化 ----
    def to_json_dict(self):
        n_in = self.n_in; h = self.h; n_out = self.n_out
        W1 = self.W1; W2 = self.W2; W3 = self.W3
        return {
            "fc1.weight": [list(W1[i*n_in:(i+1)*n_in]) for i in range(h)],
            "fc1.bias":   list(self.b1),
            "fc2.weight": [list(W2[i*h:(i+1)*h]) for i in range(h)],
            "fc2.bias":   list(self.b2),
            "fc3.weight": [list(W3[i*h:(i+1)*h]) for i in range(n_out)],
            "fc3.bias":   list(self.b3),
        }

    def from_json_dict(self, d):
        self.W1 = array('f', _flatten_2d(d["fc1.weight"]))
        self.b1 = array('f', d["fc1.bias"])
        self.W2 = array('f', _flatten_2d(d["fc2.weight"]))
        self.b2 = array('f', d["fc2.bias"])
        self.W3 = array('f', _flatten_2d(d["fc3.weight"]))
        self.b3 = array('f', d["fc3.bias"])


# -------------------- DQN Agent --------------------
class DQNAgent:
    def __init__(self, name, save_path):
        self.name = name
        self.save_path = save_path
        self.model  = QNet(STATE_SIZE, ACTION_SIZE, HIDDEN)
        self.target = QNet(STATE_SIZE, ACTION_SIZE, HIDDEN)
        self.target.copy_from(self.model)
        self.memory = []
        self.mem_idx = 0
        self.epsilon = EPSILON_START
        self.updates = 0
        self.last_loss = 0.0

    def act(self, state, valid_actions):
        if random.random() < self.epsilon:
            return valid_actions[random.randrange(len(valid_actions))]
        q = self.model.forward_q(state)
        best_a = valid_actions[0]
        best_v = q[best_a]
        for a in valid_actions:
            v = q[a]
            if v > best_v:
                best_v = v
                best_a = a
        return best_a

    def remember(self, s, a, r, ns, d):
        item = (s, a, r, ns, d)
        if len(self.memory) < MEMORY_CAPACITY:
            self.memory.append(item)
        else:
            self.memory[self.mem_idx] = item
            self.mem_idx = (self.mem_idx + 1) % MEMORY_CAPACITY

    def learn(self):
        if len(self.memory) < MIN_MEM_TO_LEARN:
            return None
        total = 0.0
        for _ in range(REPLAY_STEPS):
            s, a, r, ns, done = self.memory[random.randrange(len(self.memory))]
            if done:
                target = r
            else:
                qn = self.target.forward_q(ns)
                best = qn[0]
                for v in qn:
                    if v > best:
                        best = v
                target = r + GAMMA * best

            q, cache = self.model.forward(s)
            td = q[a] - target
            self.model.backward(cache, a, td, LR)
            total += td * td

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
        self.updates += 1
        self.last_loss = total / REPLAY_STEPS
        return self.last_loss

    def update_target(self):
        self.target.copy_from(self.model)

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
            print("  [加载] %s 解析失败: %s" % (path, e))
            return False

        if d.get("fmt") != FMT_VERSION:
            print("  [加载] fmt 不匹配"); return False
        if d.get("board") != BOARD_SIZE:
            print("  [加载] board 不匹配"); return False
        if d.get("hidden") != HIDDEN:
            print("  [加载] hidden 不匹配"); return False

        try:
            self.model.from_json_dict(d["model"])
            self.target.copy_from(self.model)
        except Exception as e:
            print("  [加载] 权重载入失败: %s" % e); return False

        self.epsilon = float(d.get("epsilon", EPSILON_START))
        self.updates = int(d.get("updates", 0))
        print("  [加载] %s 成功 (ε=%.4f, updates=%d)"
              % (path, self.epsilon, self.updates))
        return True


# -------------------- meta.json --------------------
def save_meta(path, episode, b_wins, w_wins, draws):
    with open(path, "w") as f:
        json.dump({
            "fmt": FMT_VERSION, "board": BOARD_SIZE, "hidden": HIDDEN,
            "episode": int(episode),
            "b_wins": int(b_wins), "w_wins": int(w_wins), "draws": int(draws),
        }, f)


def load_meta(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if (d.get("fmt") != FMT_VERSION or d.get("board") != BOARD_SIZE
            or d.get("hidden") != HIDDEN):
        print("  [meta] 校验失败，忽略旧进度。")
        return None
    return d


# -------------------- 单局 --------------------
def train_step(agent_b, agent_w, game):
    game.reset()
    history = []
    moves = 0
    while game.winner is None:
        p = game.current
        agent = agent_b if p == 1 else agent_w
        s = game.get_state()
        valid = game.get_valid_actions()
        if not valid:
            game.winner = 0
            break
        a = agent.act(s, valid)
        r, done = game.make_move(a)
        ns = game.get_state()
        history.append((p, s, a, r, ns, done))
        moves += 1

    winner = game.winner
    for p, s, a, r, ns, done in history:
        total = r + (1.0 if (done and winner == p) else 0.0)
        if p == 1:
            agent_b.remember(s, a, total, ns, done)
        else:
            agent_w.remember(s, a, total, ns, done)

    lb = agent_b.learn()
    lw = agent_w.learn()
    losses = [l for l in (lb, lw) if l is not None]
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return winner, moves, avg_loss


# -------------------- 主循环 --------------------
def main():
    game = GoBangGame()
    black_path = SAVE_DIR + "/black.json"
    white_path = SAVE_DIR + "/white.json"
    meta_path  = SAVE_DIR + "/meta.json"

    agent_b = DQNAgent("black", black_path)
    agent_w = DQNAgent("white", white_path)

    episode = 0
    b_wins = w_wins = draws = 0
    meta = load_meta(meta_path)
    if meta is not None:
        episode = int(meta["episode"])
        b_wins  = int(meta["b_wins"])
        w_wins  = int(meta["w_wins"])
        draws   = int(meta["draws"])
        print("[meta] 恢复进度: episode=%d, 黑胜=%d, 白胜=%d, 平=%d"
              % (episode, b_wins, w_wins, draws))

    ok_b = agent_b.load_json(black_path)
    ok_w = agent_w.load_json(white_path)
    if not (ok_b and ok_w):
        print("[提示] 权重未加载或校验失败，使用随机初始化。")

    print("=" * 62)
    print("五子棋 DQN 训练 (ESP32-S3 · 13x13 · 纯 Python)")
    print("棋盘 %dx%d | HIDDEN %d | 动作 %d | mem %d"
          % (BOARD_SIZE, BOARD_SIZE, HIDDEN, ACTION_SIZE, MEMORY_CAPACITY))
    print("存档: %s | 起始 episode=%d" % (SAVE_DIR, episode))
    print("Ctrl+C 中断并保存")
    print("=" * 62)

    t_start = time.time()
    start_ep = episode

    try:
        while True:
            winner, moves, avg_loss = train_step(agent_b, agent_w, game)
            episode += 1
            if winner == 1:
                b_wins += 1
            elif winner == 2:
                w_wins += 1
            else:
                draws += 1

            if episode % TARGET_UPDATE_INTERVAL == 0:
                agent_b.update_target()
                agent_w.update_target()

            if episode % LOG_INTERVAL == 0:
                elapsed = time.time() - t_start
                n_sess = episode - start_ep
                tot = max(b_wins + w_wins + draws, 1)
                w_str = "黑胜" if winner == 1 else ("白胜" if winner == 2 else "平局")
                print("[%6d] %s | 步数=%3d | loss=%.4f | "
                      "黑 %d(%.1f%%) 白 %d(%.1f%%) 平 %d | "
                      "ε=%.4f/%.4f | mem %d | %.1fs/ep | free %d"
                      % (episode, w_str, moves, avg_loss,
                         b_wins, 100.0*b_wins/tot,
                         w_wins, 100.0*w_wins/tot, draws,
                         agent_b.epsilon, agent_w.epsilon,
                         len(agent_b.memory),
                         elapsed/max(n_sess, 1),
                         gc.mem_free()))

            if episode % AUTO_SAVE_INTERVAL == 0:
                print("==== 自动保存 @ %d 局 ====" % episode)
                agent_b.save_json(black_path)
                agent_w.save_json(white_path)
                save_meta(meta_path, episode, b_wins, w_wins, draws)
                print("==== 保存完成 ====")

            if episode % GC_INTERVAL == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("\n手动中断，保存 @ %d 局 ..." % episode)
    except Exception as e:
        print("\n异常中断: %s" % repr(e))
    finally:
        agent_b.save_json(black_path)
        agent_w.save_json(white_path)
        save_meta(meta_path, episode, b_wins, w_wins, draws)
        print("已保存到 %s (第 %d 局)" % (SAVE_DIR, episode))


if __name__ == "__main__":
    main()
