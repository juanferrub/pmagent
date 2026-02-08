.PHONY: test evals check smoke health traces lint clean

# ─── Unit tests (fast, no network) ───────────────────────────
# Excludes test_tools.py and test_integration.py which require live APIs
FAST_TESTS = tests/ --ignore=tests/test_tools.py --ignore=tests/test_integration.py --ignore=tests/test_e2e_live.py -x --timeout=30

test:
	@echo "══════ Running unit tests ══════"
	python -m pytest $(FAST_TESTS) -v --tb=short

# ─── All tests including integration (needs live services) ───
test-all:
	@echo "══════ Running ALL tests (including integration) ══════"
	python -m pytest tests/ -v --tb=short --timeout=60

# ─── Evaluation suite ────────────────────────────────────────
evals:
	@echo "══════ Running evaluation suite ══════"
	python -m evals

# ─── Linting ─────────────────────────────────────────────────
lint:
	@echo "══════ Running linter ══════"
	python -m py_compile src/graphs/main_graph.py
	python -m py_compile src/agents/supervisor.py
	python -m py_compile src/grounding.py
	python -m py_compile src/evidence.py
	python -m py_compile src/trust_score.py
	@echo "Core modules compile OK"

# ─── Pre-commit gate (tests + lint) ─────────────────────────
check: lint test
	@echo ""
	@echo "✓ All pre-commit checks passed"

# ─── Full validation (check + evals) ────────────────────────
check-all: check evals
	@echo ""
	@echo "✓ All checks + evaluations passed"

# ─── Server smoke test (requires running server) ────────────
SERVER_URL ?= http://localhost:8001

health:
	@echo "══════ Health check ══════"
	@curl -sf $(SERVER_URL)/health | python3 -m json.tool

smoke: health
	@echo "══════ Smoke test: agent invocation ══════"
	@curl -sf -X POST $(SERVER_URL)/invoke \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer demo-token" \
		-d '{"query": "What capabilities do you have?"}' \
		| python3 -c "import sys,json; d=json.load(sys.stdin); print('Thread:', d.get('thread_id','')); print('Response:', d.get('response','')[:200])"
	@echo ""
	@echo "✓ Smoke test passed"

# ─── Verify Opik traces ─────────────────────────────────────
OPIK_URL ?= http://localhost:5174

traces:
	@echo "══════ Checking Opik traces ══════"
	@curl -sf "$(OPIK_URL)/api/v1/private/traces?project_name=pm-agent&size=3" \
		| python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Total traces: {d.get(\"total\",0)}'); [print(f'  {t[\"start_time\"][:19]}  {t.get(\"name\",\"\")}  {str(t.get(\"input\",\"\"))[:80]}') for t in d.get('content',[])[:3]]"

# ─── Start dev server ───────────────────────────────────────
serve:
	python -m uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

# ─── Cleanup ─────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned"
