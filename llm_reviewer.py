import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(diff, lint_findings, rag_chunks):
    lint_text = ""
    for f in lint_findings:
        lint_text += f"  Line {f['line']} [{f['severity']}] ({f['tool']}): {f['message']}\n"

    rag_text = ""
    for i, chunk in enumerate(rag_chunks):
        rag_text += f"--- Existing code from {chunk['filename']} (line {chunk['start_line']}) ---\n"
        rag_text += chunk['text'][:300] + "\n\n"

    prompt = f"""You are a senior software engineer doing a code review.

You are given:
1. A PR diff (the new code being added)
2. Static analysis findings from Flake8 and Bandit
3. Relevant existing code from the codebase for context

Your job is to generate clear, specific, actionable review comments.

PR DIFF:
{diff}

STATIC ANALYSIS FINDINGS:
{lint_text if lint_text else "No issues found."}

EXISTING CODEBASE CONTEXT (retrieved via RAG):
{rag_text if rag_text else "No context retrieved."}

Generate a JSON array of review comments. Each comment must have:
- "line": the line number in the diff (integer)
- "severity": one of "critical", "warning", "suggestion"
- "comment": clear explanation of the issue and how to fix it specifically for this codebase

Rules:
- critical: security vulnerabilities, bugs that will crash the code
- warning: bad practices, performance issues, code smells
- suggestion: style improvements, minor enhancements
- Be specific — reference the actual variable names and function names in the code
- If RAG context shows a better pattern already exists in the codebase, mention it explicitly
- Return ONLY the JSON array, no other text

Example format:
[
  {{
    "line": 3,
    "severity": "critical",
    "comment": "SQL injection vulnerability — never concatenate user input into queries. Use parameterized queries instead: db.query('SELECT * FROM users WHERE id=?', [user_id])"
  }}
]"""

    return prompt


def call_llm(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            comments = json.loads(raw)
            return comments

        except json.JSONDecodeError:
            print(f"  Attempt {attempt+1}: LLM returned malformed JSON, retrying...")
        except Exception as e:
            print(f"  Attempt {attempt+1}: Error — {e}, retrying...")

    print("  All retries failed. Returning empty comments.")
    return []


def review_pr(diff, lint_findings, rag_chunks):
    print("\nCalling LLM for review...")
    prompt = build_prompt(diff, lint_findings, rag_chunks)
    comments = call_llm(prompt)
    print(f"  LLM generated {len(comments)} comments")
    return comments