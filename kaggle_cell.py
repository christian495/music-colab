import os, sys, json, re, time, secrets, subprocess, urllib.request

TOPIC = "cmusic-b80fdc594f29bfe0f95c16ed"
API_KEY = secrets.token_hex(16)
PORT = 8001
REPO = "/kaggle/temp/ACE-Step-1.5"
LOG = "/kaggle/temp/install.log"
SRVLOG = "/kaggle/temp/server.log"
TUNLOG = "/kaggle/temp/tunnel.log"
RUN_ID = secrets.token_hex(4)
# v3: understands the stop command and may offer a second, high-quality server (hq_*) on the second GPU
STATE = {"stage": "start", "pct": 0, "msg": "", "url": "", "key": "", "verified": False, "v": 3, "run": RUN_ID,
         "hq": "", "hq_url": "", "hq_model": "", "hq_steps": 0, "hq_msg": ""}
srv = tun = hq_srv = hq_tun = None
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


CMD_TOPIC = TOPIC + "-cmd"
T_START = int(time.time())
_last_cmd = [0.0]
STOPPING = [False]


def stop_requested():
    """The app sends {"cmd":"stop","run":<RUN_ID>} to the -cmd topic. Polled at most every 15 seconds."""
    if STOPPING[0]:
        return True
    if time.time() - _last_cmd[0] < 15:
        return False
    _last_cmd[0] = time.time()
    try:
        raw = urllib.request.urlopen("https://ntfy.sh/%s/json?poll=1&since=%d" % (CMD_TOPIC, T_START - 30), timeout=15).read().decode("utf-8", "ignore")
    except Exception:
        return False
    for line in raw.splitlines():
        try:
            ev = json.loads(line)
            if ev.get("event") != "message":
                continue
            m = json.loads(ev.get("message", "{}"))
        except Exception:
            continue
        if m.get("cmd") == "stop" and (m.get("run") == RUN_ID or m.get("key") == API_KEY):
            STOPPING[0] = True
            return True
    return False


def kill_all():
    for p_ in (srv, tun, hq_srv, hq_tun):
        try:
            if p_ is not None:
                p_.terminate()
        except Exception:
            pass
    for _n in ("acestep-api", "cloudflared"):
        subprocess.run("pkill -f " + _n, shell=True)


def stop_now(msg="Itinigil mo ang server. Hindi na ito gumagamit ng GPU quota."):
    notify("stopped", 0, msg)
    kill_all()
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
_net_err = None
for _try in range(8):  # one slow answer from GitHub must not stop the whole start: try again for about two minutes
    try:
        urllib.request.urlopen("https://github.com", timeout=30).read(10)
        _net_err = None
        break
    except Exception as _e:
        _net_err = _e
        time.sleep(15)
if _net_err is not None:
    fail("Walang internet. Sa Settings ng notebook, i-on ang Internet. (%s)" % _net_err)

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
    # float32 by default (turbo server); the high-quality server sets ACESTEP_T4_DTYPE=bfloat16 for the big XL model
    _new = "                else:\n                    self.dtype = getattr(torch, __import__('os').environ.get('ACESTEP_T4_DTYPE', 'float32'))\n"
    open(_f, "w").write(_s.replace(_old, _new, 1))
    print("PATCHED: DiT dtype on pre-Ampere GPUs (float32 unless ACESTEP_T4_DTYPE is set)", flush=True)
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
    if stop_requested():
        stop_now()
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
    if stop_requested():
        stop_now()
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
    if stop_requested():
        stop_now()
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

# ---------- high-quality server on the second GPU (isolated: if anything fails, the fast server keeps working) ----------
HQ_PORT = 8002
HQ_LOG = "/kaggle/temp/hq_server.log"
HQ_TUNLOG = "/kaggle/temp/hq_tunnel.log"
HQ_STEPS = 50
HQ_DIRTY = [False]


def _oom_first():
    # if the machine runs out of RAM, the kernel must kill THIS process and not the fast server
    try:
        open("/proc/self/oom_score_adj", "w").write("1000")
    except Exception:
        pass


def hq_call(path, body=None, timeout=30):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (HQ_PORT, path), data=data,
        headers={"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"},
        method="POST" if body is not None else "GET")
    return urllib.request.urlopen(req, timeout=timeout).read()


def hq_try(model, dtype):
    """Start a server with this model on GPU 1 and make a 15 second test song. Returns the seconds the test song took."""
    global hq_srv
    env2 = dict(os.environ, MPLBACKEND="Agg", CUDA_VISIBLE_DEVICES="1", ACESTEP_API_KEY=API_KEY,
                ACESTEP_API_HOST="0.0.0.0", ACESTEP_API_PORT=str(HQ_PORT), ACESTEP_LM_BACKEND="pt",
                ACESTEP_INIT_LLM="false", ACESTEP_CONFIG_PATH=model, ACESTEP_T4_DTYPE=dtype)
    try:
        os.remove(HQ_LOG)
    except Exception:
        pass
    hq_srv = subprocess.Popen("uv run acestep-api --host 0.0.0.0 --port %d" % HQ_PORT, shell=True, cwd=REPO, env=env2,
                              stdout=open(HQ_LOG, "ab"), stderr=subprocess.STDOUT, preexec_fn=_oom_first)
    t0 = time.time()
    while True:
        if STOPPING[0]:
            raise Exception("itinigil")
        if hq_srv.poll() is not None:
            raise Exception("namatay ang server, exit code %s" % hq_srv.returncode)
        if time.time() - t0 > 30 * 60:
            raise Exception("masyadong matagal mag-load")
        try:
            if json.loads(hq_call("/health", timeout=5)) is not None:
                break
        except Exception:
            pass
        time.sleep(5)
    body = {"prompt": "soft acoustic pop, gentle voice", "lyrics": "[Verse]\nla la la la la\nna na na na na\n[Chorus]\nla la la la la",
            "audio_format": "mp3", "batch_size": 1, "thinking": False, "use_cot_caption": False, "use_cot_language": False,
            "audio_duration": 15, "inference_steps": HQ_STEPS, "guidance_scale": 7.0}
    r = json.loads(hq_call("/release_task", body, timeout=60))
    d = r.get("data") if isinstance(r, dict) else None
    tid = (d or {}).get("task_id") if isinstance(d, dict) else None
    if not tid:
        raise Exception("walang task id")
    t1 = time.time()
    while time.time() - t1 < 40 * 60:
        if STOPPING[0]:
            raise Exception("itinigil")
        if hq_srv.poll() is not None:
            raise Exception("namatay ang server sa test, exit code %s" % hq_srv.returncode)
        try:
            q = json.loads(hq_call("/query_result", {"task_id_list": [tid]}))
            q = q.get("data", q) if isinstance(q, dict) else q
            item = q[0] if isinstance(q, list) and q else {}
            st = int(item.get("status", 0))
        except Exception:
            st, item = 0, {}
        if st == 2:
            raise Exception("pumalpak ang test: " + str(item.get("result", ""))[:200])
        if st == 1:
            f = json.loads(item["result"])[0]["file"]
            blob = hq_call(f, timeout=120)
            if len(blob) < 60000:
                raise Exception("halos tahimik ang test na kanta (%d bytes)" % len(blob))
            return int(time.time() - t1)
        time.sleep(4)
    raise Exception("masyadong matagal ang test na kanta")


def hq_boot():
    global hq_srv, hq_tun
    try:
        gpus = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout.count("GPU ")
        if gpus < 2:
            STATE.update(hq="none", hq_msg="Isang GPU lang ang nakuha, walang mataas na kalidad na server.")
            HQ_DIRTY[0] = True
            return
        reasons = []
        for model, dtype in (("acestep-v15-xl-sft", "bfloat16"), ("acestep-v15-sft", "float32")):
            STATE.update(hq="loading", hq_msg="Nilo-load ang %s sa pangalawang GPU" % model)
            HQ_DIRTY[0] = True
            try:
                secs = hq_try(model, dtype)
                try:
                    os.remove(HQ_TUNLOG)
                except Exception:
                    pass
                hq_tun = subprocess.Popen(["cloudflared", "tunnel", "--url", "http://127.0.0.1:%d" % HQ_PORT, "--no-autoupdate"],
                                          stdout=open(HQ_TUNLOG, "ab"), stderr=subprocess.STDOUT)
                u2 = ""
                t0 = time.time()
                while not u2 and time.time() - t0 < 240:
                    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", read_all(HQ_TUNLOG))
                    if m:
                        u2 = m.group(0)
                    elif hq_tun.poll() is not None:
                        raise Exception("namatay ang tunnel")
                    else:
                        time.sleep(2)
                if not u2:
                    raise Exception("walang link ang tunnel")
                STATE.update(hq="ready", hq_url=u2, hq_model=model, hq_steps=HQ_STEPS,
                             hq_msg="%s handa. Ang 15 segundong test na kanta ay %d segundo ang tagal." % (model, secs))
                print("HQ READY:", model, dtype, secs, "s", flush=True)
                HQ_DIRTY[0] = True
                return
            except Exception as e:
                why = "%s: %s" % (model, e)
                print("HQ FAILED", why, tail(HQ_LOG, 12), flush=True)
                reasons.append(why)
                try:
                    if hq_srv is not None:
                        hq_srv.terminate()
                    subprocess.run("pkill -f 'acestep-api --host 0.0.0.0 --port %d'" % HQ_PORT, shell=True)
                except Exception:
                    pass
                if STOPPING[0]:
                    return
                time.sleep(5)
        STATE.update(hq="failed", hq_msg=("Hindi gumana ang mataas na kalidad na server. " + " | ".join(reasons))[:700])
        HQ_DIRTY[0] = True
    except Exception as e:
        STATE.update(hq="failed", hq_msg="Error sa mataas na kalidad na server: %s" % str(e)[:300])
        HQ_DIRTY[0] = True


import threading
threading.Thread(target=hq_boot, daemon=True).start()

last = time.time()
last_use = time.time()
last_jobs = -1
IDLE_LIMIT = 5 * 3600   # nobody made a song for 5 hours: stop, so the weekly GPU quota is not burned for nothing
while True:
    time.sleep(20)
    if stop_requested():
        notify("stopped", 0, "Itinigil mo ang server. Hindi na ito gumagamit ng GPU quota.", url, "", False)
        kill_all()
        break
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
        kill_all()
        break
    if srv.poll() is not None:
        notify("error", 0, "Namatay ang server. " + tail(SRVLOG))
        kill_all()
        break
    if tun.poll() is not None:
        notify("error", 0, "Namatay ang tunnel. " + tail(TUNLOG))
        kill_all()
        break
    if STATE.get("hq") == "ready" and (hq_srv is None or hq_srv.poll() is not None or hq_tun is None or hq_tun.poll() is not None):
        STATE.update(hq="failed", hq_url="", hq_msg="Namatay ang mataas na kalidad na server. Gumagana pa ang mabilis.")
        HQ_DIRTY[0] = True
    if HQ_DIRTY[0] or time.time() - last > 120:
        HQ_DIRTY[0] = False
        notify("ready", 100, "Online", url, API_KEY, True)
        last = time.time()
