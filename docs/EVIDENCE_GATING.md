# Evidence Gating System

This document describes the evidence-based reporting system that prevents hallucinations and fabricated claims in PM Agent reports.

## Overview

The PM Agent generates weekly briefings that combine data from multiple sources (Jira, GitHub, Slack, web research). Without proper safeguards, LLM-generated reports can contain fabricated claims that appear authoritative but have no basis in actual data.

The Evidence Gating System ensures that:
1. Every factual claim in a report is backed by actual tool call evidence
2. Reports with insufficient evidence are blocked from being sent
3. Missing data sources are explicitly flagged rather than fabricated
4. Human review is required for uncertain claims

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PM Agent Run                              │
├─────────────────────────────────────────────────────────────────┤
│  1. Reset Evidence Ledger                                        │
│  2. Execute tool calls (Jira, GitHub, Slack, Web)               │
│     └─> Each call recorded in Evidence Ledger                   │
│  3. Generate report draft                                        │
│  4. Pre-send Safety Gate                                         │
│     ├─> Claim Scanner validates claims against evidence         │
│     ├─> Coverage Contract checks required sources               │
│     └─> Source Validator ensures correct identifiers            │
│  5. If passed: Send email                                        │
│     If blocked: Return draft with gaps + "Needs Human Check"    │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Evidence Ledger (`src/evidence.py`)

The Evidence Ledger tracks all tool calls and their results during a run.

**What it records:**
- Source type (jira, github, slack, web, competitor)
- Tool name
- Query parameters
- Timestamp
- Result identifiers (issue keys, PR numbers, URLs, permalinks)
- Short extracted snippets (≤2 lines each)
- Success/failure status

**Example entry:**
```python
EvidenceEntry(
    id="ev-jira-0001",
    source_type=SourceType.JIRA,
    tool_name="search_jira_issues",
    query_params={"jql": "project = OPIK AND status = Open"},
    timestamp="2024-01-15T10:30:00Z",
    identifiers=["OPIK-123", "OPIK-124", "OPIK-125"],
    snippets=[
        "[OPIK-123] Fix login timeout issue",
        "[OPIK-124] Add dark mode support",
    ],
    success=True,
)
```

### 2. Claim Scanner (`src/evidence.py`)

The Claim Scanner parses report text and identifies factual claims that require evidence.

**Detected claim patterns:**
| Pattern | Claim Type | Example |
|---------|------------|---------|
| "launched", "released", "announced" | release | "LangSmith released v2.0" |
| "top N issues", "sprint velocity" | metric | "Top 5 issues this sprint" |
| "PRs merged", "tickets closed" | metric | "15 PRs merged this week" |
| "Slack highlights", "team discussions" | activity | "Slack highlights include..." |
| "blocked", "stale", "aging" | status | "3 blocked tickets" |

**Validation process:**
1. Parse report into sentences
2. Match sentences against claim patterns
3. For each claim, check if Evidence Ledger has supporting entries
4. Mark claims as verified or unverified

### 3. Coverage Contract (`src/evidence.py`)

The Coverage Contract enforces that report sections have corresponding tool calls.

**Section requirements:**
| Section | Required Source |
|---------|-----------------|
| Jira Analysis | JIRA |
| GitHub Activity | GITHUB |
| Pull Requests | GITHUB |
| Slack Highlights | SLACK |
| Competitor Updates | WEB/COMPETITOR |
| Market Trends | WEB |

**Enforcement:**
- If a section is requested but no successful tool calls exist for that source, the section is marked as uncovered
- Uncovered sections appear in the "Needs Human Check" list

### 4. Source Validator (`src/source_validation.py`)

The Source Validator ensures that identifiers in each section match the claimed source.

**Validation rules:**
- "Jira Analysis" must contain Jira issue keys (PROJ-123), not GitHub URLs
- "GitHub Activity" must contain GitHub URLs or PR numbers (#123)
- "Slack Highlights" must contain Slack permalinks or channel IDs

**Why this matters:**
Without this check, a model could claim "Jira Analysis" while citing GitHub issues, creating a confusing and potentially misleading report.

### 5. Safety Gate (`src/evidence.py`)

The Safety Gate is the final checkpoint before sending an email.

**Thresholds:**
| Metric | Threshold | Description |
|--------|-----------|-------------|
| Tool Success Rate | ≥50% | At least half of tool calls must succeed |
| Evidence Coverage | ≥70% | At least 70% of claims must have evidence |
| Section Coverage | All required | All requested sections must have data |

**If blocked:**
```json
{
  "blocked": true,
  "reason": "Tool success rate (33%) below threshold (50%)",
  "tool_success_rate": 0.33,
  "evidence_coverage": 0.5,
  "missing_sources": ["slack", "web"],
  "needs_human_check": [
    "[Jira Analysis] Sprint velocity claim - No Jira tool calls found",
    "[Data Gap] No data from slack - related sections may be incomplete"
  ],
  "action_required": "Email was NOT sent. Re-run data collection or use send_email_report_force."
}
```

## Usage

### Normal Flow

The evidence system is automatically integrated. When you invoke the graph:

```python
from src.graphs.main_graph import invoke_graph

result = await invoke_graph(
    query="Generate weekly PM briefing and email it",
    thread_id="weekly-briefing-001",
)
```

The system will:
1. Reset the evidence ledger
2. Record all tool calls via the callback handler
3. Validate the report before sending
4. Block or allow based on evidence

### Force Send (Explicit Approval)

If you need to send an incomplete report with explicit approval:

```python
# In agent conversation
send_email_report_force(
    to="pm@company.com",
    subject="Weekly Briefing (Partial)",
    body_html="<h2>Report</h2>...",
    approval_reason="User approved sending partial report due to Slack API outage"
)
```

The force-sent email will include a warning banner.

### Checking Evidence Status

To inspect the current evidence state:

```python
from src.evidence import get_ledger

ledger = get_ledger()
summary = ledger.get_coverage_summary()
print(f"Total entries: {summary['total_entries']}")
print(f"Sources covered: {summary['sources_covered']}")
```

## Escalation Behavior

When evidence is insufficient, the system escalates rather than fabricates:

### Level 1: Section Marked Unavailable
```html
<h2>Slack Highlights</h2>
<p><strong>[Data Unavailable]</strong> Slack data could not be retrieved. 
Reason: Authentication failed (not_authed)</p>
```

### Level 2: Claims Marked Uncertain
```html
<p><strong>[Unverified]</strong> Sprint velocity increased by 20%... 
(Reason: No successful Jira tool calls found)</p>
```

### Level 3: Email Blocked
```
Email was NOT sent due to insufficient evidence.

Missing sources: slack, web
Unverified claims: 5

Options:
1. Re-run data collection with proper credentials
2. Use send_email_report_force with explicit approval
3. Address the 'needs_human_check' items manually
```

## Testing

Run the evidence system tests:

```bash
pytest tests/test_evidence_system.py -v
```

Test scenarios covered:
1. **No tool calls** → System blocks send, produces draft with gaps
2. **Partial failure** → Failed source sections marked unverified
3. **Hallucinated claims** → Detected and removed/marked uncertain
4. **Successful flow** → Report with citations passes validation
5. **Source mismatch** → GitHub IDs in Jira section flagged

## Configuration

### Thresholds

Modify in `src/evidence.py`:

```python
class SafetyGate:
    MIN_TOOL_SUCCESS_RATE = 0.5  # 50%
    MIN_EVIDENCE_COVERAGE = 0.7  # 70%
```

### Claim Patterns

Add new patterns in `src/evidence.py`:

```python
EVIDENCE_REQUIRED_PATTERNS = [
    (r"\b(launched|released)\b", "release"),
    (r"\bsprint velocity\b", "metric"),
    # Add new patterns here
]
```

### Source Mappings

Update tool-to-source mappings in `src/tool_wrapper.py`:

```python
TOOL_SOURCE_MAPPING = {
    "search_jira_issues": SourceType.JIRA,
    "list_github_prs": SourceType.GITHUB,
    # Add new tools here
}
```

## Limitations

1. **Pattern-based claim detection**: May miss some claim types or have false positives
2. **Identifier extraction**: Heuristic-based, may not catch all formats
3. **No semantic understanding**: Cannot verify if a claim's meaning matches evidence
4. **Threshold tuning**: Default thresholds may need adjustment per use case

## Future Improvements

1. **Semantic claim-evidence matching**: Use embeddings to match claims to evidence
2. **Confidence scoring**: Assign confidence levels to claims based on evidence strength
3. **Audit logging**: Persistent log of all evidence decisions for compliance
4. **User preferences**: Allow users to set their own risk tolerance thresholds
