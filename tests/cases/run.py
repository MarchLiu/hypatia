#!/usr/bin/env python3
"""Run all behavioral datasets and generate a self-contained HTML report.

HYPATIA_CASE_STUB is private to child processes and runs a deterministic
SQLite FTS5 CLI stand-in in a temporary directory.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATASETS = HERE / "datasets"
TEMPLATE = HERE / "report_template.html"
DEFAULT_REPORT = HERE / "report.html"
HOOK = ROOT / "codex-integration/hooks/hypatia_user_prompt_submit.sh"
MSG_NAME = "msg-case-session-case-turn"


def read_state():
    return json.loads(Path(os.environ["HYPATIA_CASE_STATE"]).read_text())


def write_state(state):
    Path(os.environ["HYPATIA_CASE_STATE"]).write_text(json.dumps(state))


def option(args, flag, default=""):
    return args[args.index(flag) + 1] if flag in args else default


def fake_cli(args):
    state = read_state()
    command = args[0]
    if command == "knowledge-get":
        entry = next((e for e in state["entries"] if e["name"] == args[1]), None)
        print(json.dumps(entry) if entry else f"Knowledge '{args[1]}' not found.")
        return 0
    if command == "knowledge-create":
        state["entries"].append({
            "name": args[1],
            "content": {"format": "markdown", "data": option(args, "-d"),
                        "tags": option(args, "--tags").split(","),
                        "scopes": option(args, "--scopes").split(","),
                        "embed": "--no-embed" not in args},
        })
        write_state(state)
        print(f"Created knowledge: {args[1]}")
        return 0
    if command == "query":
        print("[]")
        return 0
    if command == "search":
        query = args[1]
        state["calls"].append({"command": "search", "query": query})
        write_state(state)
        db = sqlite3.connect(":memory:")
        db.execute("CREATE VIRTUAL TABLE docs_fts USING fts5(name, data, tags)")
        for entry in state["entries"]:
            content = entry["content"]
            db.execute("INSERT INTO docs_fts VALUES (?, ?, ?)",
                       (entry["name"], content["data"], " ".join(content["tags"])))
        # Mirror the English FTS5 query behavior relevant to this corpus.
        sanitized = re.sub(r"[^\w\s]", " ", query).strip()
        if not sanitized:
            print("No results found.")
            return 0
        rows = db.execute(
            "SELECT rowid, bm25(docs_fts) FROM docs_fts "
            "WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts) LIMIT ?",
            (sanitized, int(option(args, "--limit", "100"))),
        ).fetchall()
        results = []
        for rowid, rank in rows:
            entry = state["entries"][rowid - 1]
            results.append({"key": entry["name"],
                            "content": json.dumps(entry["content"]), "rank": rank})
        print(json.dumps(results) if results else "No results found.")
        return 0
    print(f"Unsupported stand-in CLI operation: {command}", file=sys.stderr)
    return 2


def check_case(case):
    with tempfile.TemporaryDirectory(prefix="hypatia-case-") as temporary:
        area = Path(temporary)
        workspace = area / case["project"]
        workspace.mkdir()
        state_path = area / "state.json"
        entries = [{"name": seed["name"], "content": {
            "format": "markdown", "data": seed["data"],
            "tags": seed.get("tags", []), "scopes": seed.get("scopes", []),
            "embed": seed.get("embed", True),
        }} for seed in case["seed"]]
        state_path.write_text(json.dumps({"entries": entries, "calls": []}))
        env = os.environ.copy()
        env.update({"HYPATIA_CASE_STUB": "1", "HYPATIA_CASE_STATE": str(state_path),
                    "HYPATIA_BIN": str(Path(__file__).resolve()),
                    "HYPO_HOME": str(area / "hook-home")})
        payload = {"hook_event_name": "UserPromptSubmit", "session_id": "case-session",
                   "turn_id": "case-turn", "cwd": str(workspace), "prompt": case["prompt"]}
        completed = subprocess.run(["/bin/bash", str(HOOK)], input=json.dumps(payload),
                                   text=True, capture_output=True, env=env, timeout=15)
        failures = []
        if completed.returncode != 0:
            return {"failures": [f"hook 退出码 {completed.returncode}: {completed.stderr.strip()}"], "observed": {}}
        try:
            output = json.loads(completed.stdout)
            context = output["hookSpecificOutput"]["additionalContext"]
        except (ValueError, KeyError) as exc:
            return {"failures": [f"hook 返回格式错误：{exc}: {completed.stdout!r}"], "observed": {}}
        recall = context.split("## Reference Information (from Hypatia)", 1)[-1] if "## Reference Information (from Hypatia)" in context else ""
        state = json.loads(state_path.read_text())
        logged = next((e for e in state["entries"] if e["name"] == MSG_NAME), None)
        if logged is None:
            return {"failures": ["当前用户消息没有写入日志"], "observed": {}}
        recalled = re.findall(r"^- ([^:]+):", recall, flags=re.MULTILINE)
        observed = {
            "recalled": recalled,
            "logged_embed": logged["content"]["embed"],
            "search_query": next((c["query"] for c in state["calls"] if c["command"] == "search"), ""),
        }
        expected = case["expect"]
        for name in expected.get("recall_contains", []):
            if f"- {name}:" not in recall:
                failures.append(f"未召回预期记忆：{name}")
        for name in expected.get("recall_excludes", []):
            if f"- {name}:" in recall:
                failures.append(f"召回了不应返回的记忆：{name}")
        if "logged_embed" in expected and logged["content"]["embed"] != expected["logged_embed"]:
            failures.append("原始消息的向量索引设置不符合预期")
        for text in expected.get("logged_excludes", []):
            if text in logged["content"]["data"]:
                failures.append("测试凭据原文被写入存储")
                observed["credential_persisted"] = True
        return {"failures": failures, "observed": observed}


def safe_prompt(prompt):
    return re.sub(r"\bsk-[A-Za-z0-9-]+\b", "sk-***", prompt)


def redact(value):
    if isinstance(value, str):
        return safe_prompt(value)
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    return value


def display_case(result):
    case = result["case"]
    expected = case["expect"]
    observed = result["observed"]
    expectations = []
    if expected.get("recall_contains"):
        expectations.append("应召回：" + ", ".join(expected["recall_contains"]))
    if expected.get("recall_excludes"):
        expectations.append("不应召回：" + ", ".join(expected["recall_excludes"]))
    if "logged_embed" in expected:
        expectations.append("原始消息可嵌入：" + ("是" if expected["logged_embed"] else "否"))
    if expected.get("logged_excludes"):
        expectations.append("保存内容应脱敏")
    actual = ["实际召回：" + (", ".join(observed.get("recalled", [])) or "无")]
    if "logged_embed" in expected:
        actual.append("原始消息可嵌入：" + ("是" if observed.get("logged_embed") else "否"))
    if observed.get("credential_persisted"):
        actual.append("测试凭据原文进入存储：是")
    diagnosis, source = case.get("diagnosis", "需要进一步定位。"), case.get("source", "")
    return {
        "id": case["id"],
        "suite": case["suite"],
        "title": case["intent"],
        "status": "fail" if result["failures"] else "pass",
        "prompt": safe_prompt(case["prompt"]),
        "expected": expectations,
        "actual": actual,
        "failures": result["failures"],
        "diagnosis": diagnosis if result["failures"] else "全部断言通过。",
        "source": source if result["failures"] else "",
        "metadata": redact({
            "repository": "MarchLiu/hypatia",
            "suite": case["suite"],
            "case_file": case["_case_file"],
            "reproduce": "python3 tests/cases/run.py --strict",
            "case": {key: value for key, value in case.items() if not key.startswith("_")},
            "observed": observed,
            "failed_assertions": result["failures"],
            "diagnosis": diagnosis if result["failures"] else "全部断言通过",
            "source": source if result["failures"] else "",
            "task": "请根据该用例定位并修复失败行为，先核对诊断与源码，再运行复现命令确认。",
            "note": "此用例只验证 hook 行为；请检查代码并复现，不要把诊断当作已证实的根因。",
        }),
    }


def render_report(results, destination):
    cases = [display_case(result) for result in results]
    for case in cases:
        if case["source"]:
            path, separator, fragment = case["source"].partition("#")
            source = (ROOT / path).resolve()
            case["source"] = os.path.relpath(source, destination.parent.resolve()) + (separator + fragment if separator else "")
    failed = sum(case["status"] == "fail" for case in cases)
    passed = len(cases) - failed
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    # JSON is embedded in a non-executing script tag; escape HTML delimiters so
    # future user-derived fixture text cannot terminate that tag.
    data = json.dumps(cases, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    document = (TEMPLATE.read_text(encoding="utf-8")
                .replace("__CASE_DATA__", data)
                .replace("__FAILED__", str(failed))
                .replace("__PASSED__", str(passed))
                .replace("__TOTAL__", str(len(cases)))
                .replace("__GENERATED_AT__", timestamp))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")


def load_cases(data_dir=DATASETS):
    controls, negatives = [], []
    seen = set()
    paths = sorted(data_dir.rglob("*.jsonl"))
    if not paths:
        raise ValueError(f"No datasets found under {data_dir}")
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
                if not isinstance(case, dict):
                    raise ValueError("each line must be a JSON object")
                required = {"id", "intent", "suite", "runner", "role", "project", "seed", "prompt", "expect"}
                missing = required - case.keys()
                if missing:
                    raise ValueError(f"missing fields: {', '.join(sorted(missing))}")
                for field in ("id", "intent", "suite", "runner", "role", "project", "prompt"):
                    if not isinstance(case[field], str) or not case[field].strip():
                        raise ValueError(f"{field} must be a nonempty string")
                if case["suite"] != path.parent.name:
                    raise ValueError("suite must match the dataset directory name")
                if case["id"] in seen:
                    raise ValueError(f"duplicate id: {case['id']}")
                if case["runner"] != "prompt_hook":
                    raise ValueError(f"unsupported runner: {case['runner']}")
                if case["role"] not in {"control", "negative"}:
                    raise ValueError(f"unsupported role: {case['role']}")
                if not isinstance(case["seed"], list):
                    raise ValueError("seed must be an array")
                for seed in case["seed"]:
                    if not isinstance(seed, dict) or not all(isinstance(seed.get(key), str) for key in ("name", "data")):
                        raise ValueError("each seed needs string name and data")
                    for key in ("scopes", "tags"):
                        if key in seed and (not isinstance(seed[key], list) or not all(isinstance(item, str) for item in seed[key])):
                            raise ValueError(f"seed {key} must be an array of strings")
                    if "embed" in seed and not isinstance(seed["embed"], bool):
                        raise ValueError("seed embed must be boolean")
                assertions = {"recall_contains", "recall_excludes", "logged_embed", "logged_excludes"}
                if not isinstance(case["expect"], dict) or not case["expect"] or set(case["expect"]) - assertions:
                    raise ValueError("expect must contain supported assertions")
                for key in ("recall_contains", "recall_excludes", "logged_excludes"):
                    if key in case["expect"] and (not isinstance(case["expect"][key], list) or not all(isinstance(item, str) for item in case["expect"][key])):
                        raise ValueError(f"expect.{key} must be an array of strings")
                if "logged_embed" in case["expect"] and not isinstance(case["expect"]["logged_embed"], bool):
                    raise ValueError("expect.logged_embed must be boolean")
                for field in ("diagnosis", "source"):
                    if field in case and not isinstance(case[field], str):
                        raise ValueError(f"{field} must be a string")
                if case.get("source") and not (ROOT / case["source"].split("#", 1)[0]).is_file():
                    raise ValueError("source file does not exist")
                case["_case_file"] = os.path.relpath(path, ROOT).replace(os.sep, "/")
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            seen.add(case["id"])
            (controls if case["role"] == "control" else negatives).append(case)
    if not controls:
        raise ValueError("At least one control case is required")
    for runner in {case["runner"] for case in negatives}:
        if not any(case["runner"] == runner for case in controls):
            raise ValueError(f"runner {runner} has no control case")
    return controls + negatives, len(controls)


def main():
    if os.environ.get("HYPATIA_CASE_STUB") == "1":
        return fake_cli(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="fail if any case fails")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT,
                        help="self-contained HTML report path")
    args = parser.parse_args()
    try:
        cases, control_count = load_cases()
    except ValueError as exc:
        print(f"Invalid test data: {exc}", file=sys.stderr)
        return 2
    failed = 0
    results = []
    for index, case in enumerate(cases):
        result = check_case(case)
        mismatches = result["failures"]
        results.append({"case": case, **result})
        print(f"{'FAIL' if mismatches else 'PASS'} {case['id']}: {case['intent']}")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
        failed += bool(mismatches)
        if index < control_count and mismatches:
            render_report(results, args.output)
            print("Control failed; the case harness is not trustworthy.", file=sys.stderr)
            return 2
    print(f"{len(cases) - failed} passed, {failed} failed, {len(cases)} total")
    render_report(results, args.output)
    print(f"HTML report: {args.output.resolve()}")
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    sys.exit(main())
