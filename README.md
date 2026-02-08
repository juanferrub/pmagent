# PM Agent - Product Management AI Agent

An AI-powered Product Management assistant built with **LangGraph** multi-agent architecture. The agent proactively monitors Slack, Jira, Notion, GitHub, and web sources to provide actionable insights, daily digests, market intelligence, and automated workflows.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Supervisor Agent                       │
│         (Intent Classification & Delegation)             │
└───────┬──────┬──────┬──────┬──────┬────────────────────┘
        │      │      │      │      │
   ┌────▼──┐ ┌▼────┐ ┌▼────┐ ┌▼────┐ ┌▼──────┐
   │ Slack │ │Jira │ │Git  │ │Mkt  │ │Notion │
   │ Agent │ │Agent│ │Agent│ │Rsch │ │Agent  │
   └───────┘ └─────┘ └─────┘ └─────┘ └───────┘
```

- **Supervisor**: Routes queries to specialist agents, aggregates results
- **Slack Agent**: Channel monitoring, message search, posting alerts
- **Jira Agent**: Issue tracking, priority management, ticket creation
- **GitHub Agent**: PR/issue monitoring, code analysis, community tracking
- **Market Research Agent**: Web search, competitor monitoring, Reddit analysis
- **Notion Agent**: Report generation, documentation, roadmap maintenance

## Quick Start

### 1. Prerequisites

- Python 3.12+
- An Anthropic API key (Claude) or OpenAI API key

### 2. Setup

```bash
# Clone and enter directory
cd pm-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run

```bash
# Start the API server
PYTHONPATH=. uvicorn api.main:app --reload --port 8000
```

### 4. Docker

```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/invoke` | Synchronous graph invocation |
| POST | `/stream` | SSE streaming responses |
| POST | `/chat` | Conversational interface |
| POST | `/webhooks/slack` | Slack event webhook |
| POST | `/webhooks/jira` | Jira event webhook |
| POST | `/webhooks/github` | GitHub event webhook |

### Example: Invoke

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"query": "Summarize the latest Jira tickets and GitHub PRs"}'
```

### Example: Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the top blockers this week?", "thread_id": "my-session"}'
```

## Proactive Schedules

| Schedule | Default | Description |
|----------|---------|-------------|
| Daily Digest | 6:00 AM CET | Slack + Jira + GitHub + Market summary |
| Weekly Market Scan | Monday 8:00 AM CET | Competitor + Reddit intelligence |
| Hourly Check | Every hour | P0/P1 ticket detection, urgent alerts |

## Environment Variables

See `.env.example` for the full list. Key variables:

- `ANTHROPIC_API_KEY` - Claude API key (required)
- `SLACK_BOT_TOKEN` - Slack bot token
- `JIRA_URL` / `JIRA_API_TOKEN` - Jira credentials
- `NOTION_API_KEY` - Notion integration token
- `GITHUB_TOKEN` - GitHub personal access token
- `TAVILY_API_KEY` - Tavily web search API key
- `POSTGRES_URI` - PostgreSQL connection string (for production checkpointing)

## Testing

```bash
# Run all tests
PYTHONPATH=. pytest tests/ -v --tb=short

# Run with coverage
PYTHONPATH=. pytest tests/ -v --cov=src --cov=api --cov-report=term-missing
```

## Trust-Critical Operations (Production-Grade Safety)

The PM Agent operates under **strict trust-critical rules** designed for production reliability. The agent is a product operations tool, not a chat assistant.

### Core Principles

1. **No Evidence → No Claim**: Facts require successful tool calls
2. **Missing Data ≠ No Problems**: Failed tools = UNKNOWN state
3. **Never Fabricate**: No invented IDs, metrics, or timelines
4. **Alerts Are Dangerous**: Only P0/P1 with full evidence
5. **One Pass, One Truth**: Each source checked once per run

### Execution Flow

Every run tracks:
```
JIRA_CHECK = NOT_STARTED → IN_PROGRESS → SUCCESS | FAILED_WITH_REASON
GITHUB_CHECK = NOT_STARTED → IN_PROGRESS → SUCCESS | FAILED_WITH_REASON
SLACK_CHECK = NOT_STARTED → IN_PROGRESS → SUCCESS | FAILED_WITH_REASON
```

No final output until all checks complete.

### Language Constraints

**Prohibited** (false reassurance):
- "Looks fine", "All good", "No major issues", "Seems okay"

**Required** (explicit uncertainty):
- "No verified critical issues detected in checked sources"
- "Data unavailable", "Unable to verify", "Unknown"

### Trust Score

Every run is scored (0-100%):

| Component | Weight | Description |
|-----------|--------|-------------|
| Evidence | 40% | Claims backed by tool calls |
| Execution | 30% | All checks completed |
| Language | 15% | No prohibited phrases |
| Alerting | 15% | Appropriate alert decisions |

### Documentation

- [docs/TRUST_CRITICAL_OPERATIONS.md](docs/TRUST_CRITICAL_OPERATIONS.md) - Full operating instructions
- [docs/EVIDENCE_GATING.md](docs/EVIDENCE_GATING.md) - Evidence system details

## Project Structure

```
pm-agent/
├── src/
│   ├── agents/           # Specialist agents (supervisor, slack, jira, etc.)
│   ├── tools/            # Integration tools (slack, jira, github, notion, research)
│   ├── graphs/           # LangGraph definitions
│   ├── state.py          # Agent state schema
│   ├── config.py         # Settings & LLM factory
│   ├── evidence.py       # Evidence Ledger & Safety Gate
│   ├── execution_state.py    # Trust-critical execution state machine
│   ├── alerting.py           # Strict alerting rules & language constraints
│   ├── trust_score.py        # Trust score calculation
│   ├── source_validation.py  # Source type validation
│   ├── evidence_callback.py  # LangChain callback for evidence recording
│   └── utils.py          # Logging, retry, circuit breaker
├── api/
│   ├── main.py           # FastAPI application
│   ├── routes.py         # API routes & webhooks
│   └── background.py     # Scheduled jobs (APScheduler)
├── tests/                # Comprehensive test suite
├── docs/
│   ├── EVIDENCE_GATING.md        # Evidence system documentation
│   └── TRUST_CRITICAL_OPERATIONS.md  # Trust-critical operating instructions
├── Dockerfile            # Multi-stage Docker build
├── docker-compose.yml    # Docker Compose with PostgreSQL
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
└── README.md             # This file
```
