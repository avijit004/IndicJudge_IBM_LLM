"""
Tiny fake OpenAI-compatible server for smoke-testing the pipeline without GPUs.
    python perturbation_experiment/tests/mock_llm_server.py 8099
Then use http://127.0.0.1:8099/v1/chat/completions as GEMMA_*_API_URL and JUDGE_API_URL.
It answers judge requests (detected by 'expert mathematical evaluator') with canned JSON.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["messages"][-1]["content"]
        if "expert mathematical evaluator" in text:
            content = json.dumps({"answer_correct": False, "reasoning_quality": 2,
                                  "follows_prompt_intent": True, "justification": "Mock judgment."})
            message = {"content": content}
        else:
            message = {"reasoning_content": "mock reasoning", "content": "mock solution. Final answer: 42"}
        out = json.dumps({"choices": [{"finish_reason": "stop", "message": message}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8099), Handler).serve_forever()
