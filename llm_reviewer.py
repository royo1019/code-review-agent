import json
import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from groq import Groq

from rag import get_ast_context

load_dotenv()

logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_prompt(diff, lint_findings, rag_chunks, ast_context: Optional[str] = None):
    """Build the LLM review prompt from diff, linter output, RAG chunks, and AST context.

    ``ast_context`` is an optional pre-formatted block describing the
    structural relationships of functions/files touched by the diff (call
    graph, import graph). When absent or empty the prompt notes that AST
    analysis is unavailable.
    """
    lint_findings = lint_findings or []
    rag_chunks = rag_chunks or []

    lint_text = ""
    for f in lint_findings:
        lint_text += f"  Line {f['line']} [{f['severity']}] ({f['tool']}): {f['message']}\n"

    rag_text = ""
    for i, chunk in enumerate(rag_chunks):
        rag_text += f"--- Existing code from {chunk['filename']} (line {chunk['start_line']}) ---\n"
        rag_text += chunk['text'][:300] + "\n\n"

    ast_block = ast_context if ast_context else "AST analysis not available."

    prompt = f"""You are a senior software engineer doing a code review.

You are given:
1. A PR diff (the new code being added)
2. Static analysis findings from Flake8 and Bandit
3. Relevant existing code from the codebase for context (retrieved via RAG)
4. Code structure analysis showing which functions the new code calls and
   which functions/files are related to it (extracted via AST parsing)

Your job is to generate clear, specific, actionable review comments.

PR DIFF:
{diff}

STATIC ANALYSIS FINDINGS:
{lint_text if lint_text else "No issues found."}

EXISTING CODEBASE CONTEXT (retrieved via RAG):
{rag_text if rag_text else "No context retrieved."}

=== CODE STRUCTURE ANALYSIS (AST) ===
{ast_block}

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
- If AST analysis shows a function already exists in the codebase that does what
  the new code is trying to do, explicitly recommend using it (cite filename and line)
- If AST analysis shows the new code calls a function that doesn't exist in the
  codebase, flag it as a likely bug (typo, missing import, or wrong name)
- If AST analysis shows the new code duplicates logic from another file, recommend
  DRY refactoring and name the existing function/file
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
    """Call the Groq LLM with retries on transient errors and malformed JSON.

    Returns an empty list after exhausting retries — never raises. Callers can
    treat ``[]`` as "no comments generated".
    """
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
    """Run a full LLM review for one PR file.

    Pulls AST structural context from the global index built by ``rag.py`` and
    passes it into ``build_prompt`` alongside the diff, lint findings, and RAG
    chunks.
    """
    print("\nCalling LLM for review...")
    try:
        ast_context = get_ast_context(diff or "")
    except Exception as e:
        logger.error(f"get_ast_context failed: {e}", exc_info=True)
        ast_context = ""

    prompt = build_prompt(diff, lint_findings, rag_chunks, ast_context)
    comments = call_llm(prompt)
    print(f"  LLM generated {len(comments)} comments")
    return comments
