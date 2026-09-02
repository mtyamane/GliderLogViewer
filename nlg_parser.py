#!/usr/bin/env python3
"""
nlg_parser.py — Slocum glider .nlg / .mlg log parser, classifier and viewer builder.

The parser reads one or more Slocum science-/network-computer log files, splits each
physical line into (timestamp, counter, message), classifies every line into a
semantic *line type*, discovers proglets and the metadata header, counts everything,
and emits a single self-contained interactive HTML report where any line type (or
proglet) can be selected to highlight every matching line in the log.

Line grammar (shared by .nlg and .mlg):

    <timestamp> [counter] <message>

    <timestamp>  float ("470.00") or right-aligned int ("  1030")
    [counter]    optional 1-4 digit sequence number, 1-2 spaces after the timestamp
    <message>    the payload, indented by 4+ spaces when no counter is present

Usage:
    python3 nlg_parser.py FILE [FILE ...] -o report.html
    python3 nlg_parser.py *.nlg -o report.html
    python3 nlg_parser.py FILE --json      # dump classified lines as JSON to stdout

The classifier is rule-based but *proglet-agnostic*: proglet names (solocam, flbbcd,
echodroid, rbrctd, rinkoII, ...) are extracted at match time, so new sensors are
handled without code changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
import re
import sys
from collections import Counter, OrderedDict

# --------------------------------------------------------------------------- #
# Line-type catalogue
# --------------------------------------------------------------------------- #
# Each type: id -> (label, category, color). Categories drive the colour family
# in the viewer; colours are the accent used for that type's rows / swatch.

TYPES: "OrderedDict[str, tuple]" = OrderedDict([
    # id                     label                              category        color
    ("meta.filename",       ("File metadata header",           "Lifecycle",    "#8c9bb0")),
    ("log.opened",          ("Log file opened",                "Lifecycle",    "#5eead4")),
    ("log.reopened",        ("Log file reopened",              "Lifecycle",    "#5eead4")),
    ("log.closed",          ("Log file closed",                "Lifecycle",    "#5eead4")),
    ("version",             ("Version banner",                 "Lifecycle",    "#8c9bb0")),
    ("time.set",            ("Time set (wall-clock)",          "Lifecycle",    "#5eead4")),

    ("proglet.start",       ("Proglet start() called",         "Proglet",      "#7ee787")),
    ("proglet.stop",        ("Proglet stop",                   "Proglet",      "#f0a05a")),

    ("behavior.state",      ("Behavior STATE change",          "Behavior",     "#7ee787")),
    ("behavior.substate",   ("Behavior SUBSTATE change",       "Behavior",     "#56d4c4")),
    ("behavior.arg",        ("Behavior argument",              "Behavior",     "#9ae6b4")),
    ("behavior.msg",        ("Behavior message",               "Behavior",     "#66b98a")),

    ("port.open",           ("Port opened",                    "Ports",        "#79c0ff")),
    ("port.config",         ("Port config (baud/framing)",     "Ports",        "#79c0ff")),
    ("port.close",          ("Port closed",                    "Ports",        "#79c0ff")),

    ("bit.raise",           ("BIT raised",                     "Built-in test","#d2a8ff")),
    ("bit.lower",           ("BIT lowered",                    "Built-in test","#d2a8ff")),
    ("bit.other",           ("BIT other",                      "Built-in test","#d2a8ff")),

    ("state.run",           ("Run-state change",               "State machine","#56d4c4")),
    ("state.surface",       ("Surface-state change",           "State machine","#56d4c4")),
    ("state.stop",          ("Stop-state change",              "State machine","#56d4c4")),

    ("device.tx",           ("Command sent  (>>)",             "Device I/O",   "#ffd166")),
    ("device.rx",           ("Device response  (<<)",          "Device I/O",   "#a5d6a7")),
    ("io.wait",             ("Response-wait window",           "Device I/O",   "#c9a15a")),

    ("file.data",           ("Data file opened/closed",        "Config",       "#a3b8ff")),
    ("xfer.timestamp",      ("Transfer timestamp",             "Config",       "#a3b8ff")),

    ("config.parsed",       ("Config value parsed",            "Config",       "#a3b8ff")),
    ("solocam.cfg",         ("Sensor cfg message",             "Config",       "#a3b8ff")),

    ("run.status",          ("Run error-count status",         "Health",       "#9db4c0")),
    ("heap.report",         ("Heap size report",               "Health",       "#9db4c0")),

    ("sensor",              ("Sensor reading",                 "Sensors",      "#c0a5f0")),
    ("gps",                 ("GPS fix / input",                "Sensors",      "#79c0ff")),
    ("comms.iridium",       ("Iridium comms",                  "Comms",        "#ffa657")),
    ("sci.relay",           ("Science relay (SCI:)",           "Comms",        "#c9a15a")),
    ("flight.control",      ("Fin / control (deadband)",       "Comms",        "#ffd166")),
    ("diag",                ("Diagnostics / device dump",      "Health",       "#9db4c0")),

    ("error",               ("ERROR",                          "Errors",       "#ff7b72")),
    ("warn",                ("Warning / timeout",              "Errors",       "#ffa657")),

    ("continuation",        ("Continuation line",              "Other",        "#6b7684")),
    ("other",               ("Unclassified",                   "Other",        "#6b7684")),
])

CATEGORY_ORDER = ["Lifecycle", "Proglet", "Behavior", "Ports", "Built-in test",
                  "State machine", "Device I/O", "Sensors", "Config", "Comms",
                  "Health", "Errors", "Other"]

# --------------------------------------------------------------------------- #
# Classification rules
# --------------------------------------------------------------------------- #
# Ordered (first match wins). Each rule: (compiled_regex, type_id, proglet_group)
# proglet_group is the regex group name that holds the proglet, or None.

_P = r"[A-Za-z][A-Za-z0-9_]*"   # proglet / identifier token

RULES = [
    # ---- flight-computer (.mlg) vocabulary ----
    (re.compile(rf"^behavior\s+(?P<proglet>{_P}):\s+STATE\b"),           "behavior.state",    "proglet"),
    (re.compile(rf"^behavior\s+(?P<proglet>{_P}):\s+SUBSTATE\b"),        "behavior.substate", "proglet"),
    (re.compile(rf"^behavior\s+(?P<proglet>{_P}):\s+(argument:|[a-z_]+\([^)]*\)\s*=)"), "behavior.arg", "proglet"),
    (re.compile(rf"^behavior\s+(?P<proglet>{_P}):"),                     "behavior.msg",      "proglet"),
    (re.compile(r"^sensor:\s*m_gps\w*"),                                "gps",           None),
    (re.compile(r"^(?:init|end)_gps_input\(\)"),                         "gps",           None),
    (re.compile(r"^(?:GPS|DR)\b.*(?:Location|Invalid|TooFar)"),          "gps",           None),
    (re.compile(r"^Waypoint:"),                                         "gps",           None),
    (re.compile(r"^sensor:\s*"),                                         "sensor",        None),
    (re.compile(r"^time set to:"),                                       "time.set",      None),
    (re.compile(r"Iridium|iridium modem|Waking up Iridium"),            "comms.iridium", None),
    (re.compile(r":OOD:|OUT OF DEADBAND"),                               "flight.control",None),
    (re.compile(r"^DRIVER_(?:ODDITY|WARNING|ERROR)"),                    "warn",          None),
    (re.compile(r"^Couldn't\b|reach or maintain commanded"),            "warn",          None),
    (re.compile(r"^(?:Changed|Restored)\s+\w+\s+from\b"),                 "sensor",        None),
    (re.compile(r"^device drivers called normally"),                    "diag",          None),
    (re.compile(r"^(?:write_segment_closing_diag_info|restore_sensors|save_and_change_sensors|print_device_queue|print_all_devices|dumped_sensors_to_file)\b"),
                                                                        "diag",          None),
    (re.compile(r"^Done:(?:print_all_devices|write_segment)"),          "diag",          None),
    (re.compile(r"^(?:Vehicle Name:|No login script|devices:\(|name\b|os w/s|Curr Time:|db\(|device scheduler|Time since device sched|>>>>)"), "diag", None),
    (re.compile(r"^#\s|^last\s+#"),                                     "diag",          None),
    (re.compile(r"^\[[IuU\- ]"),                                        "diag",          None),
    (re.compile(r"^\S+\s+[I\-]\s+[uUX\-]\b"),                           "diag",          None),
    (re.compile(r"^\S+\s+_\s+_\s+_\b"),                                 "diag",          None),
    (re.compile(r"^[A-Za-z_]\w*\s+-\s*$"),                              "diag",          None),
    # ---- science-computer (.nlg) vocabulary ----
    (re.compile(rf"^(?P<proglet>{_P})\s*<<\s"),                         "device.rx",     "proglet"),
    (re.compile(rf"^(?P<proglet>{_P})\s*>>\s"),                         "device.tx",     "proglet"),
    (re.compile(rf"^(?P<proglet>{_P})_get_response_max_wait_time\b"),   "io.wait",       "proglet"),

    (re.compile(rf"^(?:SCI:)?PROGLET\s+(?P<proglet>{_P})\s+(?:start|begin)\(\)\s+called"), "proglet.start", "proglet"),
    (re.compile(rf"^(?P<proglet>{_P})_start\s+(entered|finished)"),     "proglet.start", "proglet"),
    (re.compile(rf"^PROGLET\s+(?P<proglet>{_P})_stop\s+called"),        "proglet.stop",  "proglet"),
    (re.compile(rf"^PROGLET\s+(?P<proglet>{_P})\s+stop"),               "proglet.stop",  "proglet"),
    (re.compile(rf"^(?P<proglet>{_P})\s+stop\s+(completed|state)"),     "proglet.stop",  "proglet"),
    (re.compile(rf"^ERROR\s+from\s+PROGLET\s+(?P<proglet>{_P})"),       "error",         "proglet"),
    (re.compile(r"^ERROR\b"),                                           "error",         None),
    (re.compile(rf"PROGLET\s+(?P<proglet>{_P})_run\(\).*error"),        "error",         "proglet"),
    (re.compile(rf"^(?P<proglet>{_P})\s+Max error limit reached"),      "error",         "proglet"),
    (re.compile(rf"^(?P<proglet>{_P}),\s*state\b.*(timeout \d|no terminate|No terminate|not received|no .*response)"),
                                                                        "warn",          "proglet"),
    (re.compile(r"No .*ack received in|no .*response in|Max .*limit reached|timed out"), "warn", None),

    (re.compile(r"^(?:Opened|Closed)\s+(?P<proglet>\w+)\s+data file"),  "file.data",     "proglet"),
    (re.compile(rf"^(?P<proglet>{_P})_(get|set)_xfer_timestamp"),       "xfer.timestamp","proglet"),

    (re.compile(rf"^(?P<proglet>{_P})_change_run_state\b"),             "state.run",     "proglet"),
    (re.compile(rf"(?P<proglet>{_P})_change_surface_state\b"),          "state.surface", "proglet"),
    (re.compile(rf"^(?P<proglet>{_P})_change_stop_state\b"),            "state.stop",    "proglet"),

    (re.compile(rf"^(?P<proglet>{_P})_run\(\):.*error\(s\)"),           "run.status",    "proglet"),
    (re.compile(rf"^(?P<proglet>{_P})_run,\s*\d+:\s*state\b"),          "run.status",    "proglet"),

    (re.compile(r"^Opening\s+port\b"),                                  "port.open",     None),
    (re.compile(r"baud,\s*N81|line buf:"),                              "port.config",   None),
    (re.compile(rf"^sci_uart_close\("),                                 "port.close",    None),

    (re.compile(r"^bit_raise\b|raise count is now|Raising bit"),        "bit.raise",     None),
    (re.compile(r"^bit_lower\b|is remaining raised|is still in use|use count is now"), "bit.lower", None),
    (re.compile(r"^bit_close\b|^Bit\(|^bit_"),                          "bit.other",     None),

    (re.compile(r"^=====\s*Parsed\b"),                                  "config.parsed", None),
    (re.compile(rf"^(?P<proglet>[A-Z][A-Z0-9]{{2,}})\s+(cfg|retrieve|configuration)"), "solocam.cfg", "proglet"),
    (re.compile(r"^SOLOCAM\b"),                                         "solocam.cfg",   None),

    (re.compile(r"^report_heap_size\(\)|^SCI_M_[A-Z_]*FREE_HEAP"),      "heap.report",   None),

    (re.compile(r"LOG FILE OPENED"),                                    "log.opened",    None),
    (re.compile(r"LOG FILE REOPENED"),                                  "log.reopened",  None),
    (re.compile(r"LOG FILE CLOSED"),                                    "log.closed",    None),
    (re.compile(r"^Version\b"),                                         "version",       None),
    (re.compile(r"^SCI:"),                                              "sci.relay",     None),
]

# Prefix: timestamp, optional counter, message.
_PREFIX = re.compile(r"^(?P<ts>\s*\d+(?:\.\d+)?)(?P<gap>\s+)(?P<rest>.*)$")
_COUNTER = re.compile(r"^(?P<counter>\d{1,4})\s+(?P<msg>.*)$")
_HEADER = re.compile(r"^(?P<key>the8x3_filename|full_filename):\s*(?P<val>.+?)\s*$")


def classify(msg: str):
    """Return (type_id, proglet_or_None, actor_full_or_None) for a message body.

    For behavior lines the returned proglet is the *base* behavior name
    (surface_4 -> surface) for grouping; actor_full keeps the full instance.
    """
    for rx, tid, pg in RULES:
        m = rx.search(msg)
        if m:
            proglet = None
            if pg and pg in m.groupdict():
                proglet = m.group(pg)
                if proglet and proglet.lower() in ("the", "log", "opening", "version"):
                    proglet = None
                elif proglet:
                    proglet = proglet.lower()
            actor_full = proglet
            if tid.startswith("behavior") and proglet:
                actor_full = proglet
                proglet = re.sub(r"_\d+$", "", proglet)
            return tid, proglet, actor_full
    return "other", None, None


# non-actor flight lines still get a subsystem grouping key
_SUBSYS = {"gps": "gps", "sensor": "sensors", "comms.iridium": "iridium",
           "flight.control": "fin", "diag": "diag"}


def _fmt_dur(secs):
    """Format a duration in seconds as 'Hh MMm SSs'."""
    if secs is None:
        return "n/a"
    s = int(round(secs))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


# ---- solocam / echodroid command-type facet (Solo AUV serial protocol) ----
SOLOCAM_CMD = {"s": "single image", "c": "timelapse", "b": "burst", "v": "video clips",
    "a": "camera settings", "e": "exposure", "t": "set clock", "f": "file manager", "r": "status",
    "d": "device manager", "u": "UV lamp", "p": "parameters", "x": "shutdown", "w": "internet",
    "n": "upload script", "z": "format", "m": "metrics proc", "k": "metrics xfer",
    "g": "start-up", "i": "error"}
ECHODROID_CMD = {"telem": "telemetry", "metrics": "metrics", "stop": "stop", "rdy": "ready"}
ERR_CODES = {1: "No USB detected", 2: "USB not mounted", 3: "failed to create config file",
    4: "failed loading settings from config", 5: "failed to cast downlink telegram as ascii",
    6: "unable to parse downlink telegram", 7: "improper downlink telegram structure",
    8: "failed to set system time", 9: "improper parameter values", 10: "failed to update settings",
    11: "failed to update config file", 12: "failed to set strobe pin", 13: "failed to set pwm signal",
    14: "failed to set camera settings", 15: "failed to set exposure settings", 16: "failed to apply overlay",
    17: "failed to start preview", 18: "failed to capture image", 19: "failed to count and size files",
    20: "failed to finalize state of camera", 21: "failed to get system information",
    22: "failed to de-initialize camera", 23: "available space below threshold",
    24: "failed to write task completed message", 25: "perpetual capture error",
    26: "failed to check/increment directory", 27: "failed to print task completed message",
    28: "failed to upload new firmware", 29: "project file not found", 30: "failed to format storage device",
    31: "failed to find valid image", 32: "failed to process with soloAUVprocessing",
    33: "no metrics file found", 34: "metrics file too large, will slice to fit",
    35: "nrt metrics file too large to send", 36: "not enough storage to continue processing",
    37: "'m' image interval coerced to integer"}

_SOLO_CMD_RE = re.compile(r"'([#$]),([A-Za-z])")
_ECHO_CMD_RE = re.compile(r"'(\w+)")
_SOLO_ERR_RE = re.compile(r"'#,I,\s*(\d+)")


def extract_command(r):
    """Attach solocam/echodroid command-type facet; 'I' becomes an error."""
    if r["type"] not in ("device.tx", "device.rx"):
        return
    if r["proglet"] == "solocam":
        m = _SOLO_CMD_RE.search(r["msg"])
        if not m:
            return
        pre, letter = m.group(1), m.group(2)
        r["cmd"] = letter.lower()
        r["cmd_name"] = SOLOCAM_CMD.get(r["cmd"], r["cmd"])
        r["cmd_role"] = "command" if pre == "$" else ("ack" if letter.islower() else "response")
        if r["cmd"] == "i":
            r["type"] = "error"
            em = _SOLO_ERR_RE.search(r["msg"])
            if em:
                r["err_code"] = int(em.group(1))
                r["cmd_name"] = ERR_CODES.get(r["err_code"], "error " + em.group(1))
            else:
                r["cmd_name"] = "error"
    elif r["proglet"] == "echodroid":
        m = _ECHO_CMD_RE.search(r["msg"])
        if not m:
            return
        r["cmd"] = m.group(1).lower()
        r["cmd_name"] = ECHODROID_CMD.get(r["cmd"], r["cmd"])
        r["cmd_role"] = "send" if r["type"] == "device.tx" else "recv"


def parse_line(raw: str):
    """Split one physical line into (timestamp, counter, message)."""
    m = _PREFIX.match(raw)
    if not m:
        return None, None, raw.strip()
    ts = m.group("ts").strip()
    gap = m.group("gap")
    rest = m.group("rest")
    counter = None
    msg = rest
    if len(gap) <= 2:
        cm = _COUNTER.match(rest)
        if cm:
            counter = cm.group("counter")
            msg = cm.group("msg")
    return ts, counter, msg


def parse_file(path: str) -> dict:
    """Parse a single log file into a structured record."""
    with open(path, "r", errors="replace") as fh:
        raw_lines = fh.read().split("\n")

    meta = {"the8x3_filename": None, "full_filename": None}
    records = []           # list of line dicts
    line_no = 0

    for raw in raw_lines:
        if raw == "":
            continue
        line_no += 1

        # header metadata (first lines of the file)
        hm = _HEADER.match(raw)
        if hm:
            meta[hm.group("key")] = hm.group("val")
            records.append({
                "n": line_no, "ts": None, "counter": None,
                "msg": raw.strip(), "type": "meta.filename", "proglet": None, "behavior": None,
            })
            continue

        # dangling closing-quote continuation of a >> / << line
        if raw.strip() == "'":
            if records:
                prev = records[-1]
                prev["msg"] = prev["msg"].rstrip("\\r\\n").rstrip() + "'"
                # keep a record so line numbers stay faithful, marked continuation
            records.append({
                "n": line_no, "ts": None, "counter": None,
                "msg": "'", "type": "continuation", "proglet": None, "behavior": None,
            })
            continue

        ts, counter, msg = parse_line(raw)
        tid, proglet, actor_full = classify(msg)
        if proglet is None and tid in _SUBSYS:
            proglet = _SUBSYS[tid]
        rec = {
            "n": line_no, "ts": ts, "counter": counter,
            "msg": msg, "type": tid, "proglet": proglet, "behavior": actor_full,
            "cmd": None, "cmd_name": None, "cmd_role": None, "err_code": None,
        }
        extract_command(rec)
        records.append(rec)

    # ---- derived summaries --------------------------------------------------
    type_counts = Counter(r["type"] for r in records)
    proglet_counts = Counter(r["proglet"] for r in records if r["proglet"])

    # proglets that were actually *initiated* (start() called), in order
    initiated = []
    for r in records:
        if r["type"] == "proglet.start" and r["proglet"]:
            initiated.append({
                "proglet": r["proglet"],
                "ts": r["ts"], "counter": r["counter"], "line": r["n"],
            })

    # log kind: flight (.mlg, behaviors) vs science (.nlg, proglets)
    name = os.path.basename(path)
    if name.lower().endswith(".mlg"):
        kind = "flight"
    elif name.lower().endswith(".nlg"):
        kind = "science"
    else:
        kind = "science"
        for r in records:
            if r["type"].startswith("behavior"):
                kind = "flight"; break
            if r["type"] == "proglet.start":
                kind = "science"; break

    # flight wall-clock anchor from "time set to: <ISO>" (line has no ts of its own)
    flight_anchor = None
    last_ts = None
    for r in records:
        if r["ts"] is not None:
            try:
                last_ts = float(r["ts"])
            except ValueError:
                pass
        if r["type"] == "time.set" and flight_anchor is None:
            im = re.search(r"time set to:\s*([0-9T:\-]+)", r["msg"])
            if im:
                try:
                    dt = datetime.strptime(im.group(1), "%Y-%m-%dT%H:%M:%S")
                    epoch = int(dt.replace(tzinfo=timezone.utc).timestamp())
                    flight_anchor = {"rel": last_ts, "epoch": epoch}
                except ValueError:
                    pass

    # profile counts: dive_to / climb_to behaviors going Active
    dives = climbs = 0
    _active = re.compile(r"STATE\s+.+?->\s*Active\b")
    for r in records:
        if r["type"] == "behavior.state" and _active.search(r["msg"]):
            if r["proglet"] == "dive_to":
                dives += 1
            elif r["proglet"] == "climb_to":
                climbs += 1

    # per-file time span (seconds since log open)
    ts_vals = [float(r["ts"]) for r in records if r["ts"] is not None]
    span_secs = (max(ts_vals) - min(ts_vals)) if ts_vals else None

    return {
        "path": path,
        "name": name,
        "kind": kind,
        "meta": meta,
        "records": records,
        "type_counts": dict(type_counts),
        "proglet_counts": dict(proglet_counts),
        "initiated": initiated,
        "flight_anchor": flight_anchor,
        "dives": dives,
        "climbs": climbs,
        "down_ups": dives + climbs,
        "span_secs": span_secs,
        "n_lines": len(records),
    }


# --------------------------------------------------------------------------- #
# HTML report generation
# --------------------------------------------------------------------------- #

def build_report(parsed_files: list, title: str = "Glider log viewer") -> str:
    # aggregate totals across all files
    agg_types = Counter()
    agg_proglets = Counter()
    for pf in parsed_files:
        agg_types.update(pf["type_counts"])
        agg_proglets.update(pf["proglet_counts"])

    type_meta = {tid: {"label": lbl, "category": cat, "color": col}
                 for tid, (lbl, cat, col) in TYPES.items()}

    payload = {
        "title": title,
        "typeMeta": type_meta,
        "categoryOrder": CATEGORY_ORDER,
        "aggTypes": dict(agg_types),
        "aggProglets": dict(agg_proglets),
        "files": [{
            "name": pf["name"],
            "meta": pf["meta"],
            "initiated": pf["initiated"],
            "nLines": pf["n_lines"],
            "typeCounts": pf["type_counts"],
            "progletCounts": pf["proglet_counts"],
            "records": pf["records"],
        } for pf in parsed_files],
    }
    data_json = json.dumps(payload, separators=(",", ":"))
    return _HTML_TEMPLATE.replace("/*__DATA__*/", data_json).replace("__TITLE__", html.escape(title))


# The HTML/CSS/JS viewer. Data is injected as JSON at /*__DATA__*/.
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{
    --abyss:#0b1220; --abyss-2:#0e1626; --panel:#111c2e; --panel-2:#152238;
    --line:#1c2c44; --line-2:#24374f;
    --ink:#e6edf6; --ink-dim:#9fb0c4; --ink-faint:#67788f;
    --signal:#ffcf5c;         /* phosphor amber — the selection signal */
    --signal-soft:rgba(255,207,92,.14);
    --mono: ui-monospace,"JetBrains Mono","SF Mono",Menlo,Consolas,monospace;
    --sans: "Inter",system-ui,-apple-system,"Segoe UI",sans-serif;
    --row-h:20px;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{
    background:
      radial-gradient(1200px 600px at 80% -10%, #14243c 0%, transparent 60%),
      var(--abyss);
    color:var(--ink); font-family:var(--sans); font-size:13px;
    display:flex; flex-direction:column; height:100vh; overflow:hidden;
  }
  a{color:inherit}
  .mono{font-family:var(--mono)}

  /* ---- top bar ---- */
  header{
    display:flex; align-items:baseline; gap:16px; flex-wrap:wrap;
    padding:12px 18px; border-bottom:1px solid var(--line);
    background:linear-gradient(180deg,var(--abyss-2),transparent);
  }
  header h1{
    margin:0; font-size:14px; font-weight:600; letter-spacing:.14em;
    text-transform:uppercase; color:var(--ink);
  }
  header h1 .dot{color:var(--signal)}
  header .sub{font-size:11px; color:var(--ink-faint); letter-spacing:.04em}
  header .stat{font-family:var(--mono); font-size:11px; color:var(--ink-dim)}
  header .stat b{color:var(--signal); font-weight:600}
  .grow{flex:1 1 auto}

  /* ---- layout ---- */
  .body{display:flex; flex:1 1 auto; min-height:0}
  aside{
    width:330px; flex:0 0 330px; border-right:1px solid var(--line);
    overflow-y:auto; background:var(--abyss-2);
  }
  aside::-webkit-scrollbar,.logwrap::-webkit-scrollbar{width:10px;height:10px}
  aside::-webkit-scrollbar-thumb,.logwrap::-webkit-scrollbar-thumb{background:var(--line-2);border-radius:6px}
  main{flex:1 1 auto; display:flex; flex-direction:column; min-width:0}

  .sect{padding:14px 16px; border-bottom:1px solid var(--line)}
  .sect h2{
    margin:0 0 10px; font-size:10px; font-weight:700; letter-spacing:.16em;
    text-transform:uppercase; color:var(--ink-faint);
    display:flex; align-items:center; gap:8px;
  }
  .sect h2 .count{margin-left:auto; color:var(--ink-dim); font-family:var(--mono); letter-spacing:0}

  /* file picker */
  select.file{
    width:100%; background:var(--panel); color:var(--ink); border:1px solid var(--line-2);
    border-radius:7px; padding:8px 10px; font-family:var(--mono); font-size:12px;
  }
  .metagrid{margin-top:10px; font-family:var(--mono); font-size:11px; line-height:1.7}
  .metagrid div{display:flex; gap:8px}
  .metagrid .k{color:var(--ink-faint); flex:0 0 96px}
  .metagrid .v{color:var(--ink-dim); word-break:break-all}

  /* proglet chips */
  .chips{display:flex; flex-wrap:wrap; gap:6px}
  .chip{
    font-family:var(--mono); font-size:11px; padding:4px 9px; border-radius:20px;
    border:1px solid var(--line-2); background:var(--panel); color:var(--ink-dim);
    cursor:pointer; user-select:none; transition:.12s; white-space:nowrap;
  }
  .chip:hover{border-color:var(--ink-faint); color:var(--ink)}
  .chip.on{background:var(--signal-soft); border-color:var(--signal); color:var(--signal)}
  .chip .n{color:var(--ink-faint); margin-left:5px}
  .chip.on .n{color:var(--signal)}
  .chip.init{position:relative}
  .chip.init::before{content:"▶"; font-size:8px; margin-right:5px; color:var(--ink-faint)}
  .chip.on.init::before{color:var(--signal)}

  /* type list */
  .cat{margin-bottom:6px}
  .cat .catname{
    font-size:9.5px; letter-spacing:.14em; text-transform:uppercase;
    color:var(--ink-faint); margin:10px 0 4px;
  }
  .type{
    display:flex; align-items:center; gap:9px; padding:5px 8px; border-radius:6px;
    cursor:pointer; user-select:none; transition:.1s;
  }
  .type:hover{background:var(--panel)}
  .type.on{background:var(--signal-soft); box-shadow:inset 0 0 0 1px var(--signal)}
  .type .sw{width:9px; height:9px; border-radius:2px; flex:0 0 9px}
  .type .lbl{flex:1 1 auto; font-size:12px; color:var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  .type.on .lbl{color:var(--signal)}
  .type .ct{font-family:var(--mono); font-size:11px; color:var(--ink-faint)}
  .type.on .ct{color:var(--signal)}
  .type.zero{opacity:.32}

  /* toolbar */
  .toolbar{
    display:flex; align-items:center; gap:10px; padding:9px 14px;
    border-bottom:1px solid var(--line); background:var(--abyss-2); flex-wrap:wrap;
  }
  .search{
    display:flex; align-items:center; gap:7px; background:var(--panel);
    border:1px solid var(--line-2); border-radius:7px; padding:6px 10px; min-width:230px;
  }
  .search input{
    background:none; border:none; outline:none; color:var(--ink);
    font-family:var(--mono); font-size:12px; width:100%;
  }
  .search svg{opacity:.5}
  .toggle{
    font-size:11px; color:var(--ink-dim); display:flex; align-items:center; gap:6px;
    cursor:pointer; user-select:none; padding:6px 10px; border-radius:7px;
    border:1px solid var(--line-2); background:var(--panel);
  }
  .toggle.on{color:var(--signal); border-color:var(--signal); background:var(--signal-soft)}
  .selinfo{font-family:var(--mono); font-size:11px; color:var(--ink-dim)}
  .selinfo b{color:var(--signal)}
  .btn{
    font-size:11px; color:var(--ink-dim); cursor:pointer; padding:6px 10px;
    border-radius:7px; border:1px solid var(--line-2); background:var(--panel);
  }
  .btn:hover{color:var(--ink); border-color:var(--ink-faint)}

  /* log */
  .logwrap{flex:1 1 auto; overflow:auto; background:var(--abyss); position:relative}
  .log{font-family:var(--mono); font-size:12px; line-height:var(--row-h); min-width:max-content}
  .row{
    display:flex; align-items:baseline; gap:0; padding:0 14px 0 0;
    white-space:pre; border-left:2px solid transparent;
  }
  .row:hover{background:var(--panel)}
  .row .ln{
    flex:0 0 62px; text-align:right; padding-right:12px; color:var(--ink-faint);
    opacity:.55; user-select:none;
  }
  .row .ts{flex:0 0 76px; text-align:right; padding-right:10px; color:var(--ink-faint)}
  .row .ctr{flex:0 0 34px; text-align:right; padding-right:10px; color:#6f86ff}
  .row .msg{flex:1 1 auto; color:var(--ink); white-space:pre-wrap; word-break:break-word}
  .row .msg .pg{color:var(--signal); opacity:.9}         /* proglet name emphasis */

  /* selection highlight via one dynamic stylesheet (fast, no per-node work) */
  .log.filtering .row{display:none}
  .log.filtering .row.hit{display:flex}
  .log.dim .row:not(.hit){opacity:.16}
  .row.hit{border-left-color:var(--signal); background:var(--signal-soft)}
  .row.hit:hover{background:rgba(255,207,92,.2)}
  mark{background:var(--signal); color:#0b1220; border-radius:2px; padding:0 1px}

  .empty{padding:40px; text-align:center; color:var(--ink-faint); font-family:var(--mono); font-size:12px}
  .hint{color:var(--ink-faint); font-size:10.5px; margin-top:8px; line-height:1.5}
  kbd{font-family:var(--mono); background:var(--panel); border:1px solid var(--line-2);
      border-radius:4px; padding:1px 5px; font-size:10px; color:var(--ink-dim)}
</style>
</head>
<body>
<header>
  <h1><span class="dot">◊</span> Glider Log Viewer</h1>
  <span class="sub" id="fileSub"></span>
  <span class="grow"></span>
  <span class="stat">lines <b id="stTotal">0</b></span>
  <span class="stat">types <b id="stTypes">0</b></span>
  <span class="stat">proglets <b id="stProg">0</b></span>
</header>

<div class="body">
  <aside>
    <div class="sect">
      <h2>Log file</h2>
      <select class="file" id="fileSel"></select>
      <div class="metagrid" id="meta"></div>
    </div>

    <div class="sect">
      <h2>Proglets <span class="count" id="progCount"></span></h2>
      <div class="chips" id="progChips"></div>
      <div class="hint">▶ = initiated via <span class="mono">start()</span> in this file. Click to filter lines by proglet.</div>
    </div>

    <div class="sect">
      <h2>Line types <span class="count" id="typeCount"></span></h2>
      <div id="typeList"></div>
    </div>
  </aside>

  <main>
    <div class="toolbar">
      <label class="search">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg>
        <input id="search" placeholder="search messages…" spellcheck="false" autocomplete="off">
      </label>
      <span class="toggle" id="soloBtn" title="Show only selected lines">solo</span>
      <span class="toggle on" id="dimBtn" title="Dim non-selected lines">dim others</span>
      <span class="btn" id="clearBtn">clear selection</span>
      <span class="grow"></span>
      <span class="selinfo" id="selInfo">no selection</span>
    </div>
    <div class="logwrap" id="logwrap">
      <div class="log dim" id="log"></div>
    </div>
  </main>
</div>

<style id="hlStyle"></style>
<script>
const DATA = /*__DATA__*/;

const $ = s => document.querySelector(s);
const el = (t,c,txt)=>{const e=document.createElement(t); if(c)e.className=c; if(txt!=null)e.textContent=txt; return e;};

let fileIdx = 0;
let selTypes = new Set();      // active type ids
let selProglets = new Set();   // active proglet names
let solo = false;
let dim = true;
let query = "";

// ---- static header stats (aggregate) ----
$("#stTotal").textContent = DATA.files.reduce((a,f)=>a+f.nLines,0).toLocaleString();
$("#stTypes").textContent = Object.keys(DATA.aggTypes).length;
$("#stProg").textContent  = Object.keys(DATA.aggProglets).length;

// ---- file selector ----
const fileSel = $("#fileSel");
DATA.files.forEach((f,i)=>{
  const o = el("option", null, `${f.name}  ·  ${f.nLines.toLocaleString()} lines`);
  o.value = i; fileSel.appendChild(o);
});
fileSel.onchange = () => { fileIdx = +fileSel.value; renderAll(); };

// ---- build proglet chips + type list + meta for current file ----
function currentFile(){ return DATA.files[fileIdx]; }

function renderMeta(){
  const f = currentFile();
  $("#fileSub").textContent = f.meta.full_filename ? ("· " + f.meta.full_filename) : "";
  const m = $("#meta"); m.innerHTML = "";
  const rows = [
    ["8x3 name", f.meta.the8x3_filename || "—"],
    ["full name", f.meta.full_filename || "—"],
    ["lines", f.nLines.toLocaleString()],
    ["initiated", f.initiated.length ? [...new Set(f.initiated.map(x=>x.proglet.toLowerCase()))].join(", ") : "—"],
  ];
  rows.forEach(([k,v])=>{
    const d = el("div");
    d.appendChild(el("span","k",k));
    d.appendChild(el("span","v",v));
    m.appendChild(d);
  });
}

function renderProglets(){
  const f = currentFile();
  const wrap = $("#progChips"); wrap.innerHTML = "";
  const initiatedSet = new Set(f.initiated.map(x=>x.proglet));
  const names = Object.keys(f.progletCounts).sort();
  $("#progCount").textContent = names.length;
  if(!names.length){ wrap.appendChild(el("span","hint","none in this file")); return; }
  names.forEach(name=>{
    const c = el("span","chip"+(initiatedSet.has(name)?" init":""));
    c.appendChild(el("span",null,name));
    c.appendChild(el("span","n",f.progletCounts[name]));
    if(selProglets.has(name)) c.classList.add("on");
    c.onclick = ()=>{ toggle(selProglets,name); c.classList.toggle("on"); apply(); };
    wrap.appendChild(c);
  });
}

function renderTypes(){
  const f = currentFile();
  const list = $("#typeList"); list.innerHTML = "";
  const present = Object.keys(f.typeCounts).filter(t=>f.typeCounts[t]>0);
  $("#typeCount").textContent = present.length;

  // group by category, preserve catalogue order within category
  const byCat = {};
  for(const tid of Object.keys(DATA.typeMeta)){
    const meta = DATA.typeMeta[tid];
    const cnt = f.typeCounts[tid]||0;
    if(cnt===0) continue;
    (byCat[meta.category] ||= []).push({tid,meta,cnt});
  }
  DATA.categoryOrder.forEach(cat=>{
    if(!byCat[cat]) return;
    const box = el("div","cat");
    box.appendChild(el("div","catname",cat));
    byCat[cat].forEach(({tid,meta,cnt})=>{
      const row = el("div","type"+(selTypes.has(tid)?" on":""));
      row.dataset.tid = tid;
      const sw = el("span","sw"); sw.style.background = meta.color;
      row.appendChild(sw);
      row.appendChild(el("span","lbl",meta.label));
      row.appendChild(el("span","ct",cnt.toLocaleString()));
      row.onclick = ()=>{ toggle(selTypes,tid); row.classList.toggle("on"); apply(); };
      box.appendChild(row);
    });
    list.appendChild(box);
  });
}

// ---- render the log body for the current file ----
function esc(s){ return s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

function renderLog(){
  const f = currentFile();
  const log = $("#log");
  const html = [];
  for(const r of f.records){
    const meta = DATA.typeMeta[r.type] || DATA.typeMeta["other"];
    const pg = r.proglet ? ` data-pg="${r.proglet}"` : "";
    // emphasise the proglet token inside the message
    let msg = esc(r.msg);
    if(r.proglet){
      msg = msg.replace(new RegExp("^("+r.proglet.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")+")\\b","i"),
                        '<span class="pg">$1</span>');
    }
    html.push(
      `<div class="row" data-t="${r.type}"${pg} data-n="${r.n}">`+
      `<span class="ln">${r.n}</span>`+
      `<span class="ts">${r.ts??""}</span>`+
      `<span class="ctr">${r.counter??""}</span>`+
      `<span class="msg">${msg}</span>`+
      `</div>`
    );
  }
  log.innerHTML = html.join("");
}

// ---- selection application (fast: toggle classes + one dynamic stylesheet) ----
function toggle(set,v){ set.has(v)?set.delete(v):set.add(v); }

function apply(){
  const log = $("#log");
  const hasSel = selTypes.size || selProglets.size || query;
  const rows = log.children;

  // build match predicate
  const q = query.toLowerCase();
  let hitCount = 0;
  for(const row of rows){
    let ok = true;
    if(selTypes.size)     ok = ok && selTypes.has(row.dataset.t);
    if(selProglets.size)  ok = ok && selProglets.has(row.dataset.pg);
    if(q){
      const txt = row.lastChild.textContent.toLowerCase();
      ok = ok && txt.includes(q);
    }
    row.classList.toggle("hit", ok && hasSel);
    if(ok && hasSel) hitCount++;
  }

  log.classList.toggle("filtering", solo && hasSel);
  log.classList.toggle("dim", dim && !solo && hasSel);

  // search highlight (mark) — only when a query is present, only on hits
  if(q){ highlightMatches(q); } else { clearMarks(); }

  // selection colour follows first selected type, else signal amber
  const style = $("#hlStyle");
  let accent = "var(--signal)";
  if(selTypes.size){
    const first = [...selTypes][0];
    accent = (DATA.typeMeta[first]||{}).color || accent;
  }
  style.textContent =
    `.row.hit{border-left-color:${accent};background:${hexToSoft(accent)}}`+
    `.row.hit .ln{opacity:.9}`;

  // toolbar readout
  const info = $("#selInfo");
  if(!hasSel){ info.textContent = "no selection"; }
  else{
    const parts = [];
    if(selTypes.size) parts.push(`${selTypes.size} type${selTypes.size>1?"s":""}`);
    if(selProglets.size) parts.push(`${selProglets.size} proglet${selProglets.size>1?"s":""}`);
    if(query) parts.push(`“${query}”`);
    info.innerHTML = `<b>${hitCount.toLocaleString()}</b> line${hitCount!==1?"s":""} match · ${parts.join(" · ")}`;
  }
}

function hexToSoft(c){
  if(c.startsWith("var")) return "var(--signal-soft)";
  const h = c.replace("#",""); const n = parseInt(h.length===3?h.split("").map(x=>x+x).join(""):h,16);
  const r=(n>>16)&255,g=(n>>8)&255,b=n&255;
  return `rgba(${r},${g},${b},.14)`;
}

let markedRows = [];
function clearMarks(){
  for(const row of markedRows){
    const m = row.lastChild;
    if(m.dataset.orig!=null){ m.innerHTML = m.dataset.orig; delete m.dataset.orig; }
  }
  markedRows = [];
}
function highlightMatches(q){
  clearMarks();
  const log = $("#log");
  for(const row of log.children){
    if(!row.classList.contains("hit")) continue;
    const m = row.lastChild;
    const txt = m.textContent;
    const i = txt.toLowerCase().indexOf(q);
    if(i<0) continue;
    m.dataset.orig = m.innerHTML;
    const before = esc(txt.slice(0,i));
    const hit = esc(txt.slice(i,i+q.length));
    const after = esc(txt.slice(i+q.length));
    m.innerHTML = before + "<mark>" + hit + "</mark>" + after;
    markedRows.push(row);
    if(markedRows.length>4000) break; // safety cap for huge result sets
  }
}

// ---- toolbar wiring ----
$("#soloBtn").onclick = ()=>{ solo=!solo; $("#soloBtn").classList.toggle("on",solo);
  if(solo){ dim=false; $("#dimBtn").classList.remove("on"); } apply(); };
$("#dimBtn").onclick = ()=>{ dim=!dim; $("#dimBtn").classList.toggle("on",dim);
  if(dim){ solo=false; $("#soloBtn").classList.remove("on"); } apply(); };
$("#clearBtn").onclick = ()=>{
  selTypes.clear(); selProglets.clear(); $("#search").value=""; query="";
  document.querySelectorAll(".type.on,.chip.on").forEach(e=>e.classList.remove("on"));
  apply();
};
let searchTimer;
$("#search").oninput = e=>{ query=e.target.value.trim();
  clearTimeout(searchTimer); searchTimer=setTimeout(apply,120); };

document.addEventListener("keydown", e=>{
  if(e.key==="/" && document.activeElement.id!=="search"){ e.preventDefault(); $("#search").focus(); }
  if(e.key==="Escape"){ $("#clearBtn").click(); $("#search").blur(); }
});

// ---- full re-render on file change ----
function renderAll(){
  selTypes.clear(); selProglets.clear();
  renderMeta(); renderProglets(); renderTypes(); renderLog(); apply();
}
renderAll();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description="Parse Slocum glider .nlg/.mlg logs and build an interactive HTML viewer.")
    ap.add_argument("files", nargs="+", help="one or more .nlg / .mlg files")
    ap.add_argument("-o", "--out", default="glider_log_report.html", help="output HTML path")
    ap.add_argument("--json", action="store_true", help="print classified records as JSON to stdout instead of building HTML")
    ap.add_argument("--title", default="Glider log viewer")
    args = ap.parse_args(argv)

    parsed = []
    for path in args.files:
        if not os.path.isfile(path):
            print(f"skip (not a file): {path}", file=sys.stderr); continue
        parsed.append(parse_file(path))

    if not parsed:
        print("no files parsed", file=sys.stderr); return 1

    if args.json:
        out = [{"file": p["name"], "kind": p["kind"], "meta": p["meta"],
                "initiated": p["initiated"], "flight_anchor": p["flight_anchor"],
                "dives": p["dives"], "climbs": p["climbs"], "down_ups": p["down_ups"],
                "span_secs": p["span_secs"],
                "type_counts": p["type_counts"], "records": p["records"]} for p in parsed]
        print(json.dumps(out, indent=2))
        return 0

    html_doc = build_report(parsed, title=args.title)
    with open(args.out, "w") as fh:
        fh.write(html_doc)

    # console summary
    agg = Counter()
    for p in parsed:
        agg.update(p["type_counts"])
    print(f"Parsed {len(parsed)} file(s), {sum(p['n_lines'] for p in parsed):,} lines.")

    # per-file total time + profiles (dives / climbs / down-ups)
    print("\nPer-file summary:")
    for p in parsed:
        dur = _fmt_dur(p["span_secs"])
        prof = (f"{p['dives']} dives / {p['climbs']} climbs / {p['down_ups']} down-ups"
                if p["kind"] == "flight" else "n/a")
        print(f"  {p['name']:<28} {dur:>12}   profiles: {prof}")

    print("\nLine types found: " + str(len([t for t in agg if agg[t]])))
    for tid, cnt in agg.most_common():
        lbl = TYPES.get(tid, (tid,))[0]
        print(f"  {cnt:8,}  {lbl}")

    # command-type breakdown (solocam / echodroid I/O)
    cmd_agg = {}
    for p in parsed:
        for r in p["records"]:
            if r.get("cmd"):
                key = (r["proglet"], r["cmd"])
                cmd_agg[key] = cmd_agg.get(key, 0) + 1
    if cmd_agg:
        print("\nCommand types:")
        for (prog, cmd), cnt in sorted(cmd_agg.items(), key=lambda kv: (-kv[1], kv[0])):
            name = (SOLOCAM_CMD if prog == "solocam" else ECHODROID_CMD).get(cmd, "")
            print(f"  {cnt:8,}  {prog}:{cmd}  {name}")

    print(f"\nReport written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
