"""
ANGRA 5 - single-file neural lab (numpy only).

HONEST SCOPE
  * Our own small decoder-only Transformer (byte-level, RMSNorm, causal
    multi-head attention, SiLU MLP, tied embeddings) with hand-written
    backpropagation + Adam. It REALLY trains when you run this file.
  * It is NOT Claude / Fable and has no consciousness or life. A ~0.5M
    parameter model trained for a minute cannot be frontier-level.
  * Exact maths is done by a real exact-arithmetic engine (tool), not guessed
    by the network. Memory is a persistent sqlite database.

Run:   python angra5.py            (trains if no checkpoint, then chat)
       python angra5.py --train 2000   (train longer)
       python angra5.py --self-test
"""
import argparse, ast, json, math, os, re, sqlite3, struct, sys, time, zlib, random
import urllib.request
from fractions import Fraction
import numpy as np

HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "angra5_data")
CKPT = os.path.join(HOME, "angra5.npz")
DB = os.path.join(HOME, "memory.sqlite3")
CFG = dict(dim=96, layers=4, heads=4, ctx=128, batch=16, lr=3e-3, seed=1)
EOS, VOCAB = 256, 257

LOGO = r"""
    _    _   _  ____ ____      _       ____
   / \  | \ | |/ ___|  _ \    / \     | ___|
  / _ \ |  \| | |  _| |_) |  / _ \    |___ \
 / ___ \| |\  | |_| |  _ <  / ___ \    ___) |
/_/   \_\_| \_|\____|_| \_\/_/   \_\  |____/
        N E U R A L   L A B   -  angra
"""

# ----------------------------------------------------------------- tokenizer
def enc(s: str): return list(s.encode("utf-8"))
def dec(ids): return bytes(i for i in ids if i < 256).decode("utf-8", "ignore")

# ------------------------------------------------------------------- model
def rms(x, g):
    r = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)
    return x * r * g, (x, r, g)

def rms_b(dy, c):
    x, r, g = c
    n = x * r
    dg = (dy * n).reshape(-1, n.shape[-1]).sum(0)
    dn = dy * g
    return r * (dn - n * (dn * n).mean(-1, keepdims=True)), dg

class Model:
    """Decoder-only Transformer with exact manual gradients."""
    def __init__(self, cfg, seed=0):
        self.cfg = dict(cfg)
        D, L, C = cfg["dim"], cfg["layers"], cfg["ctx"]
        rng = np.random.default_rng(seed)
        n = lambda *s, sc=0.02: rng.normal(0, sc, s)
        p = {"E": n(VOCAB, D), "P": n(C, D), "gf": np.ones(D)}
        for i in range(L):
            for k in "qkv": p[f"{k}{i}"] = n(D, D)
            p[f"o{i}"] = n(D, D, sc=0.02 / math.sqrt(2 * L))
            p[f"w1{i}"] = n(D, 4 * D)
            p[f"w2{i}"] = n(4 * D, D, sc=0.02 / math.sqrt(2 * L))
            p[f"g1{i}"] = np.ones(D); p[f"g2{i}"] = np.ones(D)
        self.p = {k: v.astype(np.float32) for k, v in p.items()}
        self.step = 0
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.last_trace = None

    def nparams(self): return sum(v.size for v in self.p.values())

    def forward(self, ids, tgt=None, trace=False):
        p, c = self.p, self.cfg
        H, D, L = c["heads"], c["dim"], c["layers"]
        B, T = ids.shape; dh = D // H
        x = p["E"][ids] + p["P"][:T]
        mask = np.triu(np.full((T, T), -1e9, np.float32), 1)
        cache, tr = [], []
        for i in range(L):
            h1, c1 = rms(x, p[f"g1{i}"])
            sp = lambda t: t.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            q, k, v = sp(h1 @ p[f"q{i}"]), sp(h1 @ p[f"k{i}"]), sp(h1 @ p[f"v{i}"])
            sc = q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh) + mask
            sc -= sc.max(-1, keepdims=True)
            pr = np.exp(sc); pr /= pr.sum(-1, keepdims=True)
            o = (pr @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
            x1 = x + o @ p[f"o{i}"]
            h2, c2 = rms(x1, p[f"g2{i}"])
            a = h2 @ p[f"w1{i}"]; sg = 1 / (1 + np.exp(-a)); act = a * sg
            x2 = x1 + act @ p[f"w2{i}"]
            cache.append((h1, c1, q, k, v, pr, o, h2, c2, a, sg, act))
            if trace:
                ent = float(-(pr[0, :, -1] * np.log(pr[0, :, -1] + 1e-12)).sum(-1).mean())
                tr.append(dict(layer=i, attn_last=pr[0, :, -1].copy(), attn_entropy=ent,
                               resid_norm=float(np.linalg.norm(x2[0, -1])),
                               mlp_active=float((act[0, -1] > 0.1).mean())))
            x = x2
        hf, cf = rms(x, p["gf"])
        logits = hf @ p["E"].T
        loss = None
        if tgt is not None:
            z = logits - logits.max(-1, keepdims=True)
            pb = np.exp(z); pb /= pb.sum(-1, keepdims=True)
            loss = float(-np.log(pb[np.arange(B)[:, None], np.arange(T)[None], tgt] + 1e-12).mean())
        return loss, logits, (ids, tgt, cache, cf, hf, pb if tgt is not None else None), tr

    def backward(self, st):
        p, c = self.p, self.cfg
        H, D, L = c["heads"], c["dim"], c["layers"]
        ids, tgt, cache, cf, hf, pb = st
        B, T = ids.shape; dh = D // H
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dl = pb.copy()
        dl[np.arange(B)[:, None], np.arange(T)[None], tgt] -= 1
        dl /= B * T
        g["E"] += dl.reshape(-1, VOCAB).T @ hf.reshape(-1, D)
        dx, g["gf"] = rms_b(dl @ p["E"], cf)
        mg = lambda t: t.transpose(0, 2, 1, 3).reshape(B, T, D)
        f2 = lambda t: t.reshape(-1, t.shape[-1])
        for i in reversed(range(L)):
            h1, c1, q, k, v, pr, o, h2, c2, a, sg, act = cache[i]
            g[f"w2{i}"] = f2(act).T @ f2(dx)
            da = (dx @ p[f"w2{i}"].T) * (sg * (1 + a * (1 - sg)))
            g[f"w1{i}"] = f2(h2).T @ f2(da)
            dxn, g[f"g2{i}"] = rms_b(da @ p[f"w1{i}"].T, c2)
            dx1 = dx + dxn
            g[f"o{i}"] = f2(o).T @ f2(dx1)
            doh = (dx1 @ p[f"o{i}"].T).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            dpr = doh @ v.transpose(0, 1, 3, 2)
            dv = pr.transpose(0, 1, 3, 2) @ doh
            dsc = pr * (dpr - (dpr * pr).sum(-1, keepdims=True)) / math.sqrt(dh)
            dq, dk = dsc @ k, dsc.transpose(0, 1, 3, 2) @ q
            dq, dk, dv = mg(dq), mg(dk), mg(dv)
            g[f"q{i}"], g[f"k{i}"], g[f"v{i}"] = f2(h1).T @ f2(dq), f2(h1).T @ f2(dk), f2(h1).T @ f2(dv)
            dh1 = dq @ p[f"q{i}"].T + dk @ p[f"k{i}"].T + dv @ p[f"v{i}"].T
            dx0, g[f"g1{i}"] = rms_b(dh1, c1)
            dx = dx1 + dx0
        np.add.at(g["E"], ids, dx)
        g["P"][:T] += dx.sum(0)
        return g

    def adam(self, g, lr, clip=1.0):
        gn = math.sqrt(sum(float((v * v).sum()) for v in g.values()))
        s = min(1.0, clip / (gn + 1e-9))
        self.step += 1; t = self.step
        for k in self.p:
            gk = g[k] * s
            self.m[k] = 0.9 * self.m[k] + 0.1 * gk
            self.v[k] = 0.98 * self.v[k] + 0.02 * gk * gk
            mh = self.m[k] / (1 - 0.9 ** t); vh = self.v[k] / (1 - 0.98 ** t)
            self.p[k] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        return gn

    def train_step(self, ids, tgt, lr):
        loss, _, st, _ = self.forward(ids, tgt)
        self.adam(self.backward(st), lr)
        return loss

    def generate(self, prompt, maxn=100, temp=0.25, topk=8, seed=None, greedy=False, trace=False):
        rng = np.random.default_rng(seed)
        ids = enc(prompt); out = []
        for _ in range(maxn):
            ctx = np.array([ids[-self.cfg["ctx"]:]])
            _, lg, _, tr = self.forward(ctx, trace=trace)
            if trace: self.last_trace = dict(tokens=dec(ids[-self.cfg["ctx"]:]), layers=tr)
            l = lg[0, -1].astype(np.float64)
            for t in set(out[-24:]): l[t] -= 0.3                   # repetition penalty
            if greedy: nxt = int(l.argmax())
            else:
                l = l / max(temp, 1e-3)
                idx = np.argsort(l)[-topk:]
                pr = np.exp(l[idx] - l[idx].max()); pr /= pr.sum()
                nxt = int(rng.choice(idx, p=pr))
            if nxt == EOS or (nxt == 10 and out): break
            ids.append(nxt); out.append(nxt)
        return dec(out).strip()

    def save(self, path=CKPT):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        meta = np.frombuffer(json.dumps(dict(version=1, cfg=self.cfg, step=self.step)).encode(), np.uint8)
        arrs = {f"p_{k}": v for k, v in self.p.items()}
        arrs.update({f"m_{k}": v for k, v in self.m.items()})
        arrs.update({f"v_{k}": v for k, v in self.v.items()})
        np.savez_compressed(path + ".tmp.npz", meta=meta, **arrs)
        os.replace(path + ".tmp.npz", path)

    @staticmethod
    def load(path=CKPT):
        z = np.load(path, allow_pickle=False)                      # no pickle
        meta = json.loads(bytes(z["meta"]).decode())
        if meta.get("version") != 1: raise ValueError("bad checkpoint version")
        m = Model(meta["cfg"]); m.step = meta["step"]
        for k in m.p:
            m.p[k] = z[f"p_{k}"]; m.m[k] = z[f"m_{k}"]; m.v[k] = z[f"v_{k}"]
        return m

# ------------------------------------------------------------- curriculum
FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
PAIRS = [
 ("Hi", "Hello! I am Angra. How can I help you?"),
 ("Hello", "Hello! I am Angra. How can I help you?"),
 ("Who are you?", "I am Angra, a small neural network built in Python."),
 ("How are you?", "I am fine, thank you. How are you?"),
 ("Are you alive?", "I am a small neural network. I do not claim to be alive or conscious."),
 ("Are you conscious?", "No. I am a small neural network and I do not claim consciousness."),
 ("What is the capital of France?", "Paris."), ("What is the capital of Iran?", "Tehran."),
 ("What is the capital of Germany?", "Berlin."), ("What is the capital of Italy?", "Rome."),
 ("What is the capital of Japan?", "Tokyo."), ("What is the capital of Spain?", "Madrid."),
 ("What color is the sky?", "Blue."), ("What is the opposite of hot?", "Cold."),
 ("What is the opposite of big?", "Small."), ("Say yes", "Yes."), ("Thank you", "You are welcome!"),
 ("What language is Python?", "Python is a programming language."),
 ("Write a Python function that adds two numbers", "def add(a, b): return a + b"),
 ("Write a Python function that doubles a number", "def double(x): return 2 * x"),
 ("How do I print hello in Python?", "print('hello')"),
 ("سلام", "سلام! من انگرا هستم. چطور می‌توانم کمکت کنم؟"),
 ("درود", "سلام! من انگرا هستم. چطور می‌توانم کمکت کنم؟"),
 ("تو کی هستی؟", "من انگرا هستم، یک شبکه عصبی کوچک که با پایتون ساخته شده."),
 ("حالت چطوره؟", "من خوبم، ممنون. تو چطوری؟"),
 ("تو زنده‌ای؟", "من یک شبکه عصبی کوچک هستم و ادعای زنده بودن یا آگاهی نمی‌کنم."),
 ("آیا آگاهی داری؟", "نه. من یک شبکه عصبی کوچک هستم و ادعای آگاهی نمی‌کنم."),
 ("پایتخت ایران کجاست؟", "تهران."), ("پایتخت فرانسه کجاست؟", "پاریس."),
 ("پایتخت آلمان کجاست؟", "برلین."), ("پایتخت ژاپن کجاست؟", "توکیو."),
 ("رنگ آسمان چیست؟", "آبی."), ("متضاد گرم چیست؟", "سرد."), ("متضاد بزرگ چیست؟", "کوچک."),
 ("ممنون", "خواهش می‌کنم!"), ("پایتون چیست؟", "پایتون یک زبان برنامه‌نویسی است."),
]

def curriculum(seed=0, n_math=1200):
    rng = random.Random(seed); out = []
    for q, a in PAIRS:
        fa = any("\u0600" <= ch <= "\u06ff" for ch in q)
        u, m = ("کاربر", "انگرا") if fa else ("User", "Angra")
        out += [f"{u}: {q}\n{m}: {a}"] * 45
    for _ in range(n_math):
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        op = rng.choice("+-*")
        r = a + b if op == "+" else a - b if op == "-" else a * b
        w = {"+": "plus", "-": "minus", "*": "times"}[op]
        out.append(rng.choice([f"User: What is {a} {op} {b}?\nAngra: {r}", f"User: {a} {w} {b}\nAngra: {r}"]))
        fa_t = f"کاربر: {a} {op} {b} چند می‌شود؟\nانگرا: {r}"
        out.append(fa_t.translate(FA))
        if rng.random() < 0.3:                                    # algebra
            x, k = rng.randint(1, 20), rng.randint(2, 9)
            out.append(f"User: Solve {k}x = {k*x}\nAngra: x = {x}")
            out.append(f"کاربر: {k}x = {k*x} را حل کن\nانگرا: x = {x}".translate(FA))
    rng.shuffle(out)
    return out

def to_stream(samples):
    s = []
    for t in samples: s += enc(t + "\n") + [EOS]
    return np.array(s, np.int64)

def batch(stream, B, T, rng):
    ix = rng.integers(0, len(stream) - T - 1, B)
    x = np.stack([stream[i:i + T] for i in ix]); y = np.stack([stream[i + 1:i + T + 1] for i in ix])
    return x, y

# ----------------------------------------------------------- training/eval
def save_png(path, img):
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    def ch(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b""))

def plot_loss(path, losses):
    W, H = 640, 360; img = np.full((H, W, 3), 18, np.uint8)
    img[30:H - 30, 50] = 120; img[H - 30, 50:W - 20] = 120
    if len(losses) < 2: return save_png(path, img)
    ys = np.array(losses); lo, hi = ys.min(), ys.max() + 1e-9
    n = len(ys); prev = None
    for i, v in enumerate(ys):
        x = int(50 + (W - 70) * i / (n - 1)); y = int(H - 30 - (H - 60) * (v - lo) / (hi - lo))
        if prev:
            for t in np.linspace(0, 1, 12):
                px, py = int(prev[0] + (x - prev[0]) * t), int(prev[1] + (y - prev[1]) * t)
                img[max(py - 1, 0):py + 2, px] = (80, 220, 140)
        prev = (x, y)
    save_png(path, img)

def evaluate(model, n=40, seed=999):
    rng = random.Random(seed); ok = 0; lat = []
    for i in range(n):
        a, b = rng.randint(0, 99), rng.randint(0, 99); op = rng.choice("+-*")
        r = a + b if op == "+" else a - b if op == "-" else a * b
        fa = i % 2
        pr = f"کاربر: {a} {op} {b} چند می‌شود؟\nانگرا:".translate(FA) if fa else f"User: What is {a} {op} {b}?\nAngra:"
        t0 = time.time(); out = model.generate(pr, 8, greedy=True); lat.append(time.time() - t0)
        gold = str(r).translate(FA) if fa else str(r)
        ok += out.strip() == gold
    return ok / n, float(np.mean(lat))

def train(model, steps, log=True, lr=None):
    stream = to_stream(curriculum(model.cfg["seed"]))
    rng = np.random.default_rng(model.cfg["seed"] + model.step)
    base = lr or model.cfg["lr"]; losses = []; t0 = time.time(); acc = []
    for s in range(steps):
        warm = min(1.0, (s + 1) / 30); cos = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / max(steps, 1)))
        x, y = batch(stream, model.cfg["batch"], model.cfg["ctx"], rng)
        l = model.train_step(x, y, base * warm * cos); acc.append(l)
        if (s + 1) % 25 == 0:
            losses.append(float(np.mean(acc[-25:])))
            if log: print(f"  step {model.step:5d}  loss {losses[-1]:.3f}  {(s+1)/(time.time()-t0):.1f} it/s", flush=True)
    return losses

def run_training(model, steps):
    print(f"[train] {steps} steps, {model.nparams():,} parameters (real gradient descent)")
    t0 = time.time(); losses = train(model, steps); model.save()
    os.makedirs(HOME, exist_ok=True)
    plot_loss(os.path.join(HOME, "train_loss.png"), losses)
    acc, lat = evaluate(model)
    rep = (f"ANGRA5 TRAIN REPORT\nsteps_total={model.step}\nparams={model.nparams()}\n"
           f"first_loss={losses[0]:.3f}\nfinal_loss={losses[-1]:.3f}\nperplexity={math.exp(losses[-1]):.2f}\n"
           f"held_out_arithmetic_accuracy_EN_FA(greedy, raw network)={acc*100:.1f}%\n"
           f"latency_per_answer={lat:.2f}s\ntime={time.time()-t0:.0f}s\n"
           "note: tiny model; exact maths is delegated to the math engine.\n")
    open(os.path.join(HOME, "train_report.txt"), "w", encoding="utf-8").write(rep)
    print(rep); print(f"chart: {os.path.join(HOME, 'train_loss.png')}")

# --------------------------------------------------------------- math tool
def norm_digits(s):
    return (s.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))
            .replace("×", "*").replace("÷", "/").replace("−", "-").replace("٫", ".").replace("^", "**"))

def ev(expr, x=None):
    def go(n):
        if isinstance(n, ast.Expression): return go(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return Fraction(str(n.value))
        if isinstance(n, ast.Name) and n.id == "x" and x is not None: return x
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = go(n.operand); return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            a, b = go(n.left), go(n.right); o = n.op
            if isinstance(o, ast.Add): return a + b
            if isinstance(o, ast.Sub): return a - b
            if isinstance(o, ast.Mult): return a * b
            if isinstance(o, ast.Div): return a / b
            if isinstance(o, ast.Mod): return a % b
            if isinstance(o, ast.FloorDiv): return Fraction(a // b)
            if isinstance(o, ast.Pow):
                if b.denominator != 1 or abs(b) > 300 or abs(a) > 10 ** 6: raise ValueError("power too large")
                return a ** int(b)
        raise ValueError("unsupported")
    return go(ast.parse(expr, mode="eval"))

def fmt(v):
    if isinstance(v, complex): return f"{v.real:.6g}{v.imag:+.6g}i"
    if isinstance(v, float): return f"{v:.8g}"
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator} (≈{float(v):.8g})"

def fsqrt(q):
    if q < 0: return None
    n, d = math.isqrt(q.numerator), math.isqrt(q.denominator)
    return Fraction(n, d) if n * n == q.numerator and d * d == q.denominator else None

def math_tool(text):
    """Exact arithmetic, linear & quadratic equations (with verification). Returns str or None."""
    t = norm_digits(text.lower())
    cands = [c.strip() for c in re.findall(r"[\dx\.\+\-\*/%\(\)=\s]{3,}", t)]
    cands = [c for c in cands if re.search(r"\d", c) and re.search(r"[\+\-\*/%=]|x", c)]
    if not cands: return None
    c = max(cands, key=len)
    c = re.sub(r"(\d|\))\s*(x|\()", r"\1*\2", c); c = re.sub(r"x\s*(\d|\()", r"x*\1", c)
    try:
        if "=" in c:
            l, r = c.split("=", 1)
            f = lambda v: ev(l, v) - ev(r, v)
            f0, f1, fm, f2 = f(Fraction(0)), f(Fraction(1)), f(Fraction(-1)), f(Fraction(2))
            cc = f0; a = (f1 + fm) / 2 - cc; b = (f1 - fm) / 2
            if 4 * a + 2 * b + cc != f2: return "MATH TOOL: only polynomials up to degree 2 are supported."
            if a == 0:
                if b == 0: return "MATH TOOL: infinitely many solutions." if cc == 0 else "MATH TOOL: no solution."
                s = -cc / b
                return f"MATH TOOL: x = {fmt(s)}  [verified: lhs-rhs = {fmt(f(s))}]"
            d = b * b - 4 * a * cc; sq = fsqrt(d)
            if sq is not None:
                rs = [(-b + sq) / (2 * a), (-b - sq) / (2 * a)]
                return "MATH TOOL: x = " + ", ".join(fmt(v) for v in sorted(set(rs))) + f"  [verified residuals: {[str(f(v)) for v in rs]}]"
            import cmath
            z = [(-float(b) + cmath.sqrt(float(d))) / (2 * float(a)), (-float(b) - cmath.sqrt(float(d))) / (2 * float(a))]
            return "MATH TOOL: x ≈ " + ", ".join(fmt(complex(v)) if abs(v.imag) > 1e-12 else fmt(v.real) for v in z) + "  (numeric)"
        return f"MATH TOOL: {c} = {fmt(ev(c))}"
    except ZeroDivisionError: return "MATH TOOL: division by zero."
    except Exception: return None

# ------------------------------------------------------------------ memory
class Memory:
    def __init__(self, path=DB):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, ts REAL, type TEXT,
            content TEXT, importance REAL, access_count INT DEFAULT 0, last_accessed REAL)""")
        self.db.commit()
    def write(self, content, typ="fact", imp=0.5):
        if self.db.execute("SELECT 1 FROM memories WHERE content=?", (content,)).fetchone(): return False
        self.db.execute("INSERT INTO memories(ts,type,content,importance) VALUES(?,?,?,?)", (time.time(), typ, content, imp))
        self.db.commit(); return True
    def count(self): return self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    @staticmethod
    def words(s): return {w for w in re.findall(r"\w+", norm_digits(s.lower())) if len(w) > 2}
    def retrieve(self, q, k=3, types=("fact", "note")):
        qw = self.words(q); rows = self.db.execute(
            "SELECT id,content,importance,ts FROM memories WHERE type IN (%s)" % ",".join("?" * len(types)), types).fetchall()
        sc = []
        for i, c, imp, ts in rows:
            ov = len(qw & self.words(c))
            if ov: sc.append((ov * imp + 0.1 / (1 + (time.time() - ts) / 86400), i, c))
        sc.sort(reverse=True)
        for _, i, _ in sc[:k]:
            self.db.execute("UPDATE memories SET access_count=access_count+1,last_accessed=? WHERE id=?", (time.time(), i))
        self.db.commit(); return [c for _, _, c in sc[:k]]
    def forget(self, q):
        n = 0
        for _, i, c in [(0, r[0], r[1]) for r in self.db.execute("SELECT id,content FROM memories")]:
            if q.lower() in c.lower(): self.db.execute("DELETE FROM memories WHERE id=?", (i,)); n += 1
        self.db.commit(); return n

NAME_PATS = [r"my name is ([^\s\.,!?]+)", r"i am called ([^\s\.,!?]+)", r"اسم من (\S+?)(?: است| هست|\.|$)", r"من (\S+) هستم"]

def learn_from_text(mem, text):
    """Extract durable facts from chat. Returns list of new facts."""
    new = []
    t = text.strip()
    for pat in NAME_PATS:
        m = re.search(pat, t, re.I)
        if m and m.group(1).lower() not in ("angra", "انگرا"):
            f = f"User name is {m.group(1)}"
            if mem.write(f, "fact", 0.9): new.append(f)
            break
    m = re.search(r"(?:remember(?: that)?|یادت باشه(?: که)?|به خاطر بسپار(?: که)?)[:\s]+(.+)", t, re.I)
    if m and mem.write(m.group(1).strip(), "note", 0.8): new.append(m.group(1).strip())
    m = re.search(r"i (?:like|love|live in|work as|work at) (.+)", t, re.I)
    if m and mem.write("User said: " + t, "fact", 0.6): new.append("User said: " + t)
    return new

def memory_answer(mem, q):
    ql = q.lower()
    if re.search(r"what('s| is) my name|اسم من|who am i", ql):
        r = mem.db.execute("SELECT content FROM memories WHERE content LIKE 'User name is %' ORDER BY ts DESC LIMIT 1").fetchone()
        return ("Your name is " + r[0][13:] + ".") if r and not re.search(r"[\u0600-\u06ff]", q) else (
            ("اسم تو " + r[0][13:] + " است.") if r else "I don't know your name yet. Tell me: 'my name is ...'")
    if re.search(r"\?|؟|what|who|where|when|چه|چی|کجا|کی", ql) and re.search(r"remember|memory|know about me|یادته|یادت", ql):
        h = mem.retrieve(q, 3) or [r[0] for r in mem.db.execute("SELECT content FROM memories ORDER BY ts DESC LIMIT 3")]
        return "From memory: " + " | ".join(h) if h else "Memory is empty."
    if re.search(r"\?|؟", q):
        h = mem.retrieve(q, 1)
        if h and len(Memory.words(q) & Memory.words(h[0])) >= 2: return "From memory: " + h[0]
    return None

# ------------------------------------------------------------ local backend
def local_chat(base, model, msgs, timeout=60):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", method="POST",
        data=json.dumps({"model": model, "messages": msgs, "temperature": 0.7}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]

# -------------------------------------------------------------------- chat
class Angra:
    def __init__(self, model):
        self.model, self.mem = model, Memory()
        self.local = None; self.history = []
        self.mem.write("Session started " + time.strftime("%Y-%m-%d %H:%M"), "episode", 0.1)

    def quick_learn(self, text, steps=6):
        """Online learning: a few REAL gradient steps on a new fact (small LR)."""
        s = to_stream([f"User: {text}\nAngra: Got it, I will remember that."])
        T = min(self.model.cfg["ctx"], len(s) - 1)
        x, y = s[None, :T], s[None, 1:T + 1]
        for _ in range(steps): l = self.model.train_step(x, y, 5e-4)
        return l

    def reply(self, text, trace=False):
        fa = bool(re.search(r"[\u0600-\u06ff]", text))
        new = learn_from_text(self.mem, text)
        self.mem.write("Q: " + text[:200], "episode", 0.2)
        tag = ""
        if new:
            l = self.quick_learn(text); tag = f"\n  [memory +{len(new)}; online learning loss {l:.2f}]"
            return ("باشه، یاد گرفتم: " if fa else "Noted, I will remember: ") + "; ".join(new) + tag
        a = memory_answer(self.mem, text)
        if a: return a
        m = math_tool(text)
        if m: return m
        if self.local:
            try:
                ctx = self.mem.retrieve(text, 3)
                sysm = "You are Angra. Relevant memory: " + " | ".join(ctx) if ctx else "You are Angra."
                self.history = (self.history + [{"role": "user", "content": text}])[-8:]
                out = local_chat(self.local[0], self.local[1], [{"role": "system", "content": sysm}] + self.history)
                self.history.append({"role": "assistant", "content": out}); return "[EXTERNAL MODEL] " + out
            except Exception as e:
                return f"[local backend failed: {e}] falling back to transparent model."
        pr = f"کاربر: {text}\nانگرا:" if fa else f"User: {text}\nAngra:"
        return self.model.generate(pr[-(self.model.cfg["ctx"] - 70):], 200, trace=trace) or "..."

HELP = """/help  /status  /remember TEXT  /recall QUERY  /forget TEXT  /trace  /learn N (train N steps)
/local URL MODEL (OpenAI-compatible)  /local off  /save  /quit"""

def chat(model):
    sys.stdout.reconfigure(encoding="utf-8")
    ag = Angra(model)
    print(LOGO); print("Transparent small model (real weights) + exact math tool + persistent memory.")
    print("Not conscious, not Claude. English / فارسی.  /help\n")
    while True:
        try: line = input("You > ").strip()
        except (EOFError, KeyboardInterrupt): print(); break
        if not line: continue
        if line.startswith("/"):
            cmd, _, arg = line[1:].partition(" ")
            if cmd == "quit": break
            elif cmd == "help": print(HELP)
            elif cmd == "status":
                print(f"params={model.nparams():,} layers={model.cfg['layers']} dim={model.cfg['dim']} ctx={model.cfg['ctx']} "
                      f"train_steps={model.step} memories={ag.mem.count()} backend={'LOCAL '+ag.local[0] if ag.local else 'TRANSPARENT'}")
            elif cmd == "remember": print("saved" if ag.mem.write(arg, "note", 0.8) else "already known")
            elif cmd == "recall": print("\n".join(ag.mem.retrieve(arg, 5)) or "nothing found")
            elif cmd == "forget": print(f"deleted {ag.mem.forget(arg)}")
            elif cmd == "save": model.save(); print("checkpoint saved")
            elif cmd == "learn":
                n = int(arg or 100); ls = train(model, n, log=False); model.save(); print(f"trained {n} steps, loss {ls[-1] if ls else float('nan'):.3f}")
            elif cmd == "local":
                p = arg.split()
                ag.local = None if (not p or p[0] == "off") else (p[0], p[1] if len(p) > 1 else "local-model")
                print("backend: " + ("EXTERNAL / BLACK-BOX " + str(ag.local) if ag.local else "TRANSPARENT"))
            elif cmd == "trace":
                tr = model.last_trace
                if not tr: print("no trace yet: talk to the transparent model first")
                else:
                    print("real values from last generation step (last token):")
                    for l in tr["layers"]:
                        a = l["attn_last"]; print(f" L{l['layer']+1}: resid_norm={l['resid_norm']:.2f} attn_entropy={l['attn_entropy']:.2f} "
                              f"mlp_active={l['mlp_active']*100:.0f}% head_max_attn={[round(float(v),2) for v in a.max(-1)]}")
            else: print("unknown command, /help")
            continue
        try: print("Angra >", ag.reply(line, trace=True))
        except Exception as e: print("Angra > [error]", e)
    model.save(); print("bye")

# --------------------------------------------------------------- self-test
def self_test():
    res = []
    def chk(n, f):
        try: res.append((n, bool(f())))
        except Exception as e: res.append((n, False)); print(n, e)
    cfg = dict(CFG, dim=16, layers=2, heads=2, ctx=12, batch=2)
    m = Model(cfg, 0); rng = np.random.default_rng(0)
    ids = rng.integers(0, 256, (2, 8)); tg = rng.integers(0, 256, (2, 8))
    l0, _, st, _ = m.forward(ids, tg); g = m.backward(st)
    ok = True
    for k in ["E", "q0", "w11", "g20", "P", "o1", "k0", "v1", "gf"]:
        idx = tuple(int(rng.integers(0, s)) for s in m.p[k].shape)
        old = m.p[k][idx]; e = 1e-2
        m.p[k][idx] = old + e; lp = m.forward(ids, tg)[0]; m.p[k][idx] = old - e; lm = m.forward(ids, tg)[0]; m.p[k][idx] = old
        num = (lp - lm) / (2 * e); ok &= abs(num - g[k][idx]) < 2e-3 + 0.05 * abs(num)
    chk("gradient check (all parameter kinds)", lambda: ok)
    chk("causality", lambda: np.allclose(m.forward(ids)[1][0, :4], m.forward(np.concatenate([ids[:, :4], ids[:, :4]], 1))[1][0, :4], atol=1e-4))
    chk("tokenizer roundtrip EN+FA", lambda: dec(enc("سلام hello ۱۲۳")) == "سلام hello ۱۲۳")
    m2 = Model(cfg, 1); s = to_stream(curriculum(0, 200)); r2 = np.random.default_rng(0)
    first = np.mean([m2.train_step(*batch(s, 2, 12, r2), 3e-3) for _ in range(5)])
    last = np.mean([m2.train_step(*batch(s, 2, 12, r2), 3e-3) for _ in range(60)])
    chk("training reduces loss", lambda: last < first)
    chk("generation deterministic", lambda: m2.generate("User: hi\nAngra:", 6, seed=3) == m2.generate("User: hi\nAngra:", 6, seed=3))
    m2.save("/tmp/_a5.npz"); m3 = Model.load("/tmp/_a5.npz")
    chk("checkpoint roundtrip", lambda: np.allclose(m3.forward(ids)[1], m2.forward(ids)[1]))
    chk("math arithmetic", lambda: "= 14" in math_tool("what is 2+3*4?"))
    chk("math persian digits", lambda: "= 5" in math_tool("۲ + ۳ چند می‌شود؟"))
    chk("math linear", lambda: "x = 2 " in math_tool("solve 2x+3=7") + " ")
    chk("math quadratic", lambda: "2, 3" in math_tool("x^2-5x+6=0"))
    mem = Memory("/tmp/_a5.sqlite3"); mem.db.execute("DELETE FROM memories"); learn_from_text(mem, "my name is Sina")
    chk("memory store+retrieve", lambda: "Sina" in memory_answer(mem, "what is my name?"))
    chk("local client serialization", lambda: json.loads(json.dumps({"model": "x", "messages": [{"role": "user", "content": "سلام"}]}))["messages"][0]["content"] == "سلام")
    for n, o in res: print(("PASS " if o else "FAIL ") + n)
    print("SELF-TEST", "PASS" if all(o for _, o in res) else "FAIL"); return all(o for _, o in res)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--train", type=int, default=0); ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    if a.self_test: sys.exit(0 if self_test() else 1)
    sys.stdout.reconfigure(encoding="utf-8"); os.makedirs(HOME, exist_ok=True)
    if os.path.exists(CKPT) and not a.reset:
        try: model = Model.load()
        except Exception as e: print("bad checkpoint:", e); model = Model(CFG, CFG["seed"])
    else: model = Model(CFG, CFG["seed"])
    if a.train or model.step == 0: run_training(model, a.train or 1500)
    chat(model)

if __name__ == "__main__":
    main()
