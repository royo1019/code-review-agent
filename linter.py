import os
import json
import subprocess
import tempfile

def save_file_to_temp(filename, content):
    temp_dir = tempfile.mkdtemp()
    filepath = os.path.join(temp_dir, os.path.basename(filename))
    with open(filepath, "w") as f:
        f.write(content)
    return filepath

# ─── Python linters ──────────────────────────────────────
def run_flake8(filepath):
    result = subprocess.run(
        ["flake8", filepath],
        capture_output=True,
        text=True
    )
    findings = []
    for line in result.stdout.strip().split("\n"):
        if line:
            parts = line.split(":")
            if len(parts) >= 4:
                findings.append({
                    "tool": "flake8",
                    "line": int(parts[1]),
                    "severity": "warning",
                    "message": parts[3].strip()
                })
    return findings

def run_bandit(filepath):
    result = subprocess.run(
        ["bandit", "-r", filepath, "-f", "json", "-q"],
        capture_output=True,
        text=True
    )
    findings = []
    try:
        data = json.loads(result.stdout)
        for issue in data.get("results", []):
            findings.append({
                "tool": "bandit",
                "line": issue["line_number"],
                "severity": issue["issue_severity"].lower(),
                "message": issue["issue_text"]
            })
    except json.JSONDecodeError:
        pass
    return findings

# ─── JavaScript / TypeScript linter ──────────────────────
def run_eslint(filepath):
    result = subprocess.run(
        ["eslint", filepath, "--format", "json", "--no-eslintrc",
         "--rule", "no-unused-vars: warn",
         "--rule", "no-undef: warn",
         "--rule", "eqeqeq: warn",
         "--rule", "no-eval: error",
         "--env", "browser,node,es6",
         "--parser-options", "ecmaVersion:2020"],
        capture_output=True,
        text=True
    )
    findings = []
    try:
        data = json.loads(result.stdout)
        for file_result in data:
            for msg in file_result.get("messages", []):
                findings.append({
                    "tool": "eslint",
                    "line": msg.get("line", 1),
                    "severity": "warning" if msg["severity"] == 1 else "critical",
                    "message": msg["message"]
                })
    except (json.JSONDecodeError, KeyError):
        pass
    return findings

# ─── Check if a tool is installed ────────────────────────
def is_tool_installed(tool):
    result = subprocess.run(
        ["which", tool],
        capture_output=True,
        text=True
    )
    return result.returncode == 0

# ─── Main router ─────────────────────────────────────────
def run_linters(filename, content):
    filepath = save_file_to_temp(filename, content)
    findings = []

    print(f"\nRunning linters on {filename}...")

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".py":
        flake8 = run_flake8(filepath)
        bandit = run_bandit(filepath)
        findings = flake8 + bandit
        print(f"  Flake8: {len(flake8)} issues")
        print(f"  Bandit: {len(bandit)} issues")

    elif ext in [".js", ".ts", ".jsx", ".tsx"]:
        if is_tool_installed("eslint"):
            eslint = run_eslint(filepath)
            findings = eslint
            print(f"  ESLint: {len(eslint)} issues")
        else:
            print("  ESLint not installed — skipping JS/TS linting")
            print("  Install with: npm install -g eslint")

    elif ext in [".java"]:
        print("  Java detected — LLM-only review (no linter configured)")

    elif ext in [".go"]:
        print("  Go detected — LLM-only review (no linter configured)")

    elif ext in [".rb"]:
        print("  Ruby detected — LLM-only review (no linter configured)")

    elif ext in [".md", ".txt", ".json", ".yaml", ".yml"]:
        print(f"  {ext} file — LLM-only review")

    else:
        print(f"  Unsupported extension {ext} — LLM-only review")

    print(f"  Total: {len(findings)} issues")
    return findings