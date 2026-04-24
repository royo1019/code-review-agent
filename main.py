import argparse
from agent import run_agent

parser = argparse.ArgumentParser(description="AI Code Review Agent")
parser.add_argument("--repo", required=True, help="GitHub repo e.g. royo1019/test-review-agent")
parser.add_argument("--pr", required=True, type=int, help="PR number e.g. 1")
parser.add_argument("--path", required=True, help="Local codebase path e.g. /Users/royo/test-repo")

args = parser.parse_args()
run_agent(args.repo, args.pr, args.path)