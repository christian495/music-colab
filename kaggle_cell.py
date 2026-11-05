import os, sys, json, re, time, secrets, subprocess, urllib.request

TOPIC = "cmusic-b80fdc594f29bfe0f95c16ed"
API_KEY = secrets.token_hex(16)
PORT = 8001
REPO = "/kaggle/temp/ACE-Step-1.5"
LOG = "/kaggle/temp/install.log"
SRVLOG = "/kaggle/temp/server.log"
TUNLOG = "/kaggle/temp/tunnel.log"
STATE = {"stage": "start", "pct": 0, "msg": "", "url": "", "key": "", "verified": False, "v": 2}
os.makedirs("/kaggle/temp", exist_ok=True)
os.environ["PATH"] = "/root/.local/bin:/usr/local/bin:" + os.environ.get("PATH", "")


def notify(stage, pct, msg, url="", key="", verified=False):
    STATE.update(stage=stage, pct=pct, msg=msg, url=url, key=key, verified=verified)
    print("[%s%%] %s: %s" % (pct, stage, msg), flush=True)
    try:
        req = urllib.request.Request(
            "https://ntfy.sh/" + TOPIC,
            data=json.dumps(STATE).encode("utf-8"),
            method="POST",
        )
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as e:
        print("notify failed:", e, flush=True)


def sh(cmd, log, cwd=None, env=None):
    with open(log, "ab") as f:
        return subprocess.run(
            cmd, shell=True, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT
        ).returncode


def tail(path, n=30):
    try:
        with open(path, errors="ignore") as f:
            return "".join(f.readlines()[-n:])[-2200:]
    except Exception:
        return ""


def read_all(path):
    try:
        with open(path, errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def diag():
    out = []
    for c in ("free -m | sed -n 2p", "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"):
        try:
            out.append(subprocess.run(c, shell=True, capture_output=True, text=True, timeout=15).stdout.strip())
        except Exception:
            pass
    return " | ".join(out)


def fail(msg, path=""):
    notify("error", 0, (msg + " " + tail(path)).strip())
    raise SystemExit(msg)


for _name in ("acestep-api", "cloudflared"):
    subprocess.run("pkill -f " + _name, shell=True)
time.sleep(2)
for _f in (SRVLOG, TUNLOG):
    try:
        os.remove(_f)
    except Exception:
        pass

notify("start", 2, "Nagsisimula na")

if sh("nvidia-smi -L", LOG) != 0:
    fail("Walang GPU. Sa Settings ng notebook, piliin ang Accelerator: GPU T4 x2.")
_gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, capture_output=True, text=True).stdout
if "P100" in _gpu:
    fail("P100 ang GPU at hindi ito tugma. Piliin ang Accelerator: GPU T4 x2.")
try:
    urllib.request.urlopen("https://github.com", timeout=15).read(10)
except Exception as _e:
    fail("Walang internet. Sa Settings ng notebook, i-on ang Internet. (%s)" % _e)

notify("install", 5, "Ini-install ang uv")
if sh("curl -LsSf https://astral.sh/uv/install.sh | sh", LOG) != 0:
    fail("Hindi na-install ang uv.", LOG)

notify("install", 10, "Dina-download ang ACE-Step")
if not os.path.isdir(REPO):
    if sh("git clone --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git " + REPO, LOG) != 0:
        fail("Hindi na-clone ang ACE-Step.", LOG)

notify("install", 15, "Ini-install ang mga package, 5 hanggang 10 minuto")
if sh("uv sync", LOG, cwd=REPO, env=dict(os.environ, MPLBACKEND="Agg")) != 0:
    fail("Pumalpak ang uv sync.", LOG)

# T4 (pre-Ampere) runs the DiT in float16, which gives NaN latents as soon as lyrics are used (instrumental worked).
# ACESTEP_DTYPE is only mentioned in the error text and is not implemented, so patch the dtype in the source.
_f = REPO + "/acestep/core/generation/handler/init_service_orchestrator.py"
_s = open(_f).read()
_old = "                else:\n                    self.dtype = torch.float16\n"
if _old in _s:
    open(_f, "w").write(_s.replace(_old, "                else:\n                    self.dtype = torch.float32\n", 1))
    print("PATCHED: DiT dtype float32 on pre-Ampere GPUs", flush=True)
else:
    print("PATCH NOT FOUND: dtype line changed upstream", flush=True)

notify("model", 40, "Sinisimulan ang server at dina-download ang model")
env = dict(
    os.environ,
    MPLBACKEND="Agg",
    CUDA_VISIBLE_DEVICES="0",
    ACESTEP_API_KEY=API_KEY,
    ACESTEP_API_HOST="0.0.0.0",
    ACESTEP_API_PORT=str(PORT),
    ACESTEP_LM_BACKEND="pt",
    ACESTEP_INIT_LLM="false",   # pure DiT mode: the 1.7B language model was loaded at startup and got the server killed (exit 137)
)
srv = subprocess.Popen(
    "uv run acestep-api --host 0.0.0.0 --port %d" % PORT,
    shell=True, cwd=REPO, env=env,
    stdout=open(SRVLOG, "ab"), stderr=subprocess.STDOUT,
)


def healthy():
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:%d/health" % PORT,
            headers={"Authorization": "Bearer " + API_KEY},
        )
        return urllib.request.urlopen(req, timeout=5).status == 200
    except Exception:
        return False


start = time.time()
last = 0
while not healthy():
    if srv.poll() is not None:
        fail("Namatay ang server habang nagsisimula. Exit code %s. %s" % (srv.returncode, diag()), SRVLOG)
    if time.time() - start > 45 * 60:
        fail("Masyadong matagal ang pag-load ng model.", SRVLOG)
    if time.time() - last > 60:
        pct = min(85, 40 + int((time.time() - start) / 60 * 3))
        notify("model", pct, "Naglo-load ng model")
        last = time.time()
    time.sleep(5)

def api_call(path, body=None, timeout=30):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (PORT, path),
        data=data,
        headers={"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "ignore"))
    d = r.get("data") if isinstance(r, dict) else None
    return d if d is not None else r


# Real test: the model only loads on the first song, so make a short one BEFORE saying "Handa na".
notify("model", 86, "Sinusubukan ang model sa maikling kanta. Ito ang unang pag-load, matagal ito")
task_id = None
try:
    r = api_call("/release_task", {"prompt": "soft acoustic pop, gentle voice", "lyrics": "[Verse]\nla la la la la\nna na na na na\n[Chorus]\nla la la la la",
                                   "audio_format": "mp3", "batch_size": 1, "thinking": False,
                                   "use_cot_caption": False, "use_cot_language": False,
                                   "audio_duration": 15})
    task_id = r.get("task_id") if isinstance(r, dict) else None
except Exception as e:
    fail("Hindi tinanggap ng server ang test na kanta: %s" % e, SRVLOG)
if not task_id:
    fail("Walang task id ang test na kanta.", SRVLOG)

t1 = time.time()
last = 0
done = False
while time.time() - t1 < 40 * 60:
    if srv.poll() is not None:
        fail("Namatay ang server habang sinusubukan ang model. Exit code %s. %s" % (srv.returncode, diag()), SRVLOG)
    st = 0
    item = None
    try:
        q = api_call("/query_result", {"task_id_list": [task_id]})
        item = q[0] if isinstance(q, list) and q else None
        st = int(item.get("status", 0)) if isinstance(item, dict) else 0
    except Exception:
        st = 0
    if st == 1:
        done = True
        break
    if st == 2:
        fail("Pumalpak ang test na kanta. " + str((item or {}).get("result", ""))[:300], SRVLOG)
    if time.time() - last > 20:
        el = int(time.time() - t1)
        notify("model", min(98, 86 + el // 30), "Naglo-load at sinusubukan ang model, %d segundo na" % el)
        last = time.time()
    time.sleep(4)
if not done:
    fail("Masyadong matagal ang test na kanta.", SRVLOG)
notify("model", 99, "Tagumpay ang test na kanta")

notify("tunnel", 99, "Ginagawa ang online na link")
if sh("curl -L -o /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /usr/local/bin/cloudflared", LOG) != 0:
    fail("Hindi na-download ang cloudflared.", LOG)
tun = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://127.0.0.1:%d" % PORT, "--no-autoupdate"],
    stdout=open(TUNLOG, "ab"), stderr=subprocess.STDOUT,
)
url = ""
t0 = time.time()
while not url and time.time() - t0 < 240:
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", read_all(TUNLOG))
    if m:
        url = m.group(0)
    elif tun.poll() is not None:
        fail("Namatay ang tunnel.", TUNLOG)
    else:
        time.sleep(2)
if not url:
    fail("Walang link na lumabas.", TUNLOG)

notify("ready", 100, "Handa na", url, API_KEY, True)
print("LINK:", url)
print("KEY:", API_KEY)

last = time.time()
last_use = time.time()
last_jobs = -1
IDLE_LIMIT = 5 * 3600   # nobody made a song for 5 hours: stop, so the weekly GPU quota is not burned for nothing
while True:
    time.sleep(20)
    try:
        _st = api_call("/v1/stats")
        _total = int(((_st or {}).get("jobs") or {}).get("total", -1))
        if _total != last_jobs:
            last_jobs = _total
            last_use = time.time()
    except Exception:
        pass
    if time.time() - last_use > IDLE_LIMIT:
        notify("stopped", 0, "Natulog ang server dahil walang gumamit ng 5 oras. Pasimulan ulit.")
        break
    if srv.poll() is not None:
        notify("error", 0, "Namatay ang server. " + tail(SRVLOG))
        break
    if tun.poll() is not None:
        notify("error", 0, "Namatay ang tunnel. " + tail(TUNLOG))
        break
    if time.time() - last > 120:
        notify("ready", 100, "Online", url, API_KEY, True)
        last = time.time()
