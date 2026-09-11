# app.py
import threading
import time
import json
import os
from collections import deque
from datetime import datetime

from flask import Flask, render_template_string, jsonify

app = Flask(__name__)

MAX_RECORDS = 20                      # 最多保存 20 次数据
STATE_FILE = "records.json"           # 想持久化就留，不想就删掉相关代码

# ---------------- 全局状态 ----------------
_lock = threading.Lock()
_history = deque(maxlen=MAX_RECORDS)  # 关键：满了自动丢最旧的
_state = {
    "running": True,
    "epoch": 0,
    "loss": None,
    "accuracy": None,
    "last_update": None,
    "message": "启动中...",
}


# ---------------- 你的训练逻辑放这里 ----------------
def train_step(epoch):
    """
    替换成你真正的训练代码。
    返回一个 dict，比如 {"loss": 0.12, "accuracy": 0.93}
    """
    time.sleep(0.5)  # 模拟一次训练耗时
    loss = round(max(0.01, 1.0 / (epoch + 1)), 4)
    acc = round(min(0.99, 0.5 + epoch * 0.02), 4)
    return {"loss": loss, "accuracy": acc}


def save_history():
    """把当前历史写盘（整体覆盖，天然只留 20 条）"""
    try:
        with _lock:
            data = list(_history)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, STATE_FILE)   # 原子替换，避免写坏
    except Exception as e:
        print("保存失败:", e)


def load_history():
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data[-MAX_RECORDS:]:
            _history.append(item)     # deque 会自动裁剪
    except Exception as e:
        print("读取失败:", e)


# ---------------- 后台训练线程 ----------------
def training_worker():
    epoch = 0
    while True:
        try:
            result = train_step(epoch)
            epoch += 1
            item = {
                "epoch": epoch,
                "loss": result.get("loss"),
                "accuracy": result.get("accuracy"),
                "time": datetime.now().strftime("%H:%M:%S"),
            }
            with _lock:
                _history.append(item)          # 超过 20 条自动覆盖最旧
                _state.update({
                    "epoch": epoch,
                    "loss": item["loss"],
                    "accuracy": item["accuracy"],
                    "last_update": item["time"],
                    "message": f"第 {epoch} 轮完成",
                })
            save_history()
        except Exception as e:
            with _lock:
                _state["message"] = f"训练出错: {e}"
            time.sleep(2)


# ---------------- 路由 ----------------
@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify({
            "state": dict(_state),
            "history": list(_history),
            "max": MAX_RECORDS,
        })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with _lock:
        _history.clear()
        _state["message"] = "历史已清空"
    save_history()
    return jsonify({"ok": True})


# ---------------- 页面 ----------------
PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>训练监控</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px;
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: #0f172a; color: #e2e8f0;
  }
  h1 { font-size: 20px; margin: 0 0 18px; }
  .cards { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }
  .card {
    flex: 1; min-width: 150px;
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 14px 16px;
  }
  .card .label { font-size: 12px; color: #94a3b8; margin-bottom: 6px; }
  .card .value { font-size: 22px; font-weight: 600; color: #38bdf8; }
  .bar {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px; font-size: 13px; color: #94a3b8;
  }
  button {
    background: #334155; color: #e2e8f0; border: none;
    padding: 7px 14px; border-radius: 6px; cursor: pointer; font-size: 13px;
  }
  button:hover { background: #475569; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { padding: 9px 12px; text-align: left; border-bottom: 1px solid #1e293b; }
  th { color: #94a3b8; font-weight: 500; font-size: 12px; }
  tbody tr:nth-child(odd) { background: #162032; }
  tbody tr.newest { background: #1e3a5f; }
  .dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #22c55e; margin-right: 6px;
    animation: pulse 1.4s infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }
</style>
</head>
<body>
  <h1><span class="dot"></span>训练监控面板</h1>

  <div class="cards">
    <div class="card"><div class="label">当前轮次</div><div class="value" id="epoch">-</div></div>
    <div class="card"><div class="label">Loss</div><div class="value" id="loss">-</div></div>
    <div class="card"><div class="label">Accuracy</div><div class="value" id="acc">-</div></div>
    <div class="card"><div class="label">最近更新</div><div class="value" id="upd">-</div></div>
  </div>

  <div class="bar">
    <span id="msg">加载中...</span>
    <span>保存记录：<b id="cnt">0</b> / 20（超出自动覆盖最旧）</span>
    <button onclick="resetAll()">清空历史</button>
  </div>

  <table>
    <thead>
      <tr><th>轮次</th><th>Loss</th><th>Accuracy</th><th>时间</th></tr>
    </thead>
    <tbody id="rows"></tbody>
  </table>

<script>
{% raw %}
async function refresh() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    const s = d.state;

    document.getElementById('epoch').textContent = s.epoch || 0;
    document.getElementById('loss').textContent  = s.loss ?? '-';
    document.getElementById('acc').textContent   = s.accuracy ?? '-';
    document.getElementById('upd').textContent   = s.last_update ?? '-';
    document.getElementById('msg').textContent   = s.message || '';
    document.getElementById('cnt').textContent   = d.history.length;

    // 倒序显示，最新的在最上面
    const rows = d.history.slice().reverse().map((it, i) => `
      <tr class="${i === 0 ? 'newest' : ''}">
        <td>${it.epoch}</td>
        <td>${it.loss ?? '-'}</td>
        <td>${it.accuracy ?? '-'}</td>
        <td>${it.time}</td>
      </tr>`).join('');
    document.getElementById('rows').innerHTML = rows || '<tr><td colspan="4">暂无数据</td></tr>';
  } catch (e) {
    document.getElementById('msg').textContent = '连接失败: ' + e;
  }
}

async function resetAll() {
  await fetch('/api/reset', { method: 'POST' });
  refresh();
}

refresh();
setInterval(refresh, 1000);
{% endraw %}
</script>
</body>
</html>
"""


# ---------------- 启动 ----------------
load_history()

# 关键：进程一启动就开训，与有没有人访问无关
threading.Thread(target=training_worker, daemon=True).start()

if __name__ == "__main__":
    # debug=True 会启动 reloader，导致训练线程跑两遍，所以关掉
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
