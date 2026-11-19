"""Agent tools for BioEvidence."""

import json
from dataclasses import dataclass

import structlog

from bioevidence.db.session import get_session
from bioevidence.llm import LLMClient, create_stub_client
from bioevidence.retrieval.pipeline import Hit, get_retriever

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """Result from search_evidence tool."""

    hits: list[Hit]
    query: str


class SearchEvidenceTool:
    """Tool for searching evidence corpus."""

    def __init__(self, system: str = "A5", llm_client: LLMClient | None = None):
        self.system = system
        self.retriever = get_retriever(system)
        self.llm = llm_client or create_stub_client()

    def execute(self, query: str, filters: dict | None = None, k: int = 5) -> SearchResult:
        """Execute search."""
        hits = self.retriever.search(query, k=k, filters=filters)
        return SearchResult(hits=hits, query=query)


class SQLQueryTool:
    """Tool for executing read-only SQL queries."""

    ALLOWED_TABLES = {
        "trials",
        "trial_conditions",
        "trial_interventions",
        "trial_outcomes",
        "pubmed_articles",
        "drug_labels",
        "chunks",
    }
    BLOCKED_FUNCTIONS = {"pg_", "information_schema", "COPY", "INTO"}
    MAX_LIMIT = 200

    def __init__(self):
        pass

    def _validate_sql(self, sql: str) -> tuple[bool, str]:
        """Basic SQL validation."""
        sql_lower = sql.lower().strip()

        # Must be SELECT only
        if not sql_lower.startswith("select"):
            return False, "Only SELECT queries allowed"

        # Check for blocked functions
        for blocked in self.BLOCKED_FUNCTIONS:
            if blocked.lower() in sql_lower:
                return False, f"Blocked function/keyword: {blocked}"

        # Check tables (basic check)
        # In production, use proper SQL parser (sqlglot)
        return True, ""

    def execute(self, sql: str) -> list[dict]:
        """Execute SQL query."""
        # Validate
        valid, error = self._validate_sql(sql)
        if not valid:
            raise ValueError(f"SQL validation failed: {error}")

        # Add LIMIT if not present
        if "limit" not in sql.lower():
            sql = f"{sql.rstrip(';')} LIMIT {self.MAX_LIMIT}"

        with get_session() as session:
            result = session.execute(sql)
            columns = result.keys()
            rows = [dict(zip(columns, row, strict=False)) for row in result.fetchall()]
            return rows


class CompareTrialsTool:
    """Tool for comparing trials side-by-side."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or create_stub_client()

    def execute(self, nct_ids: list[str]) -> dict:
        """Compare 2-4 trials."""
        if len(nct_ids) < 2 or len(nct_ids) > 4:
            raise ValueError("Must provide 2-4 NCT IDs")

        # Fetch trial data
        trials_data = {}
        with get_session() as session:
            from bioevidence.db.models import Trial, TrialCondition, TrialIntervention, TrialOutcome

            for nct_id in nct_ids:
                trial = session.get(Trial, nct_id)
                if not trial:
                    continue

                conditions = (
                    session.query(TrialCondition).filter(TrialCondition.nct_id == nct_id).all()
                )
                interventions = (
                    session.query(TrialIntervention)
                    .filter(TrialIntervention.nct_id == nct_id)
                    .all()
                )
                outcomes = session.query(TrialOutcome).filter(TrialOutcome.nct_id == nct_id).all()

                trials_data[nct_id] = {
                    "title": trial.title,
                    "phase": trial.phase,
                    "status": trial.overall_status,
                    "enrollment": trial.enrollment,
                    "start_date": trial.start_date.isoformat() if trial.start_date else None,
                    "conditions": [c.condition for c in conditions],
                    "interventions": [i.name for i in interventions],
                    "primary_outcomes": [o.measure for o in outcomes if o.kind.value == "primary"],
                    "secondary_outcomes": [
                        o.measure for o in outcomes if o.kind.value == "secondary"
                    ],
                }

        return {
            "trials": trials_data,
            "comparison": self._generate_comparison(trials_data),
        }

    def _generate_comparison(self, trials_data: dict) -> str:
        """Generate comparison summary."""
        if not trials_data:
            return "No trials found."

        lines = ["## Trial Comparison\n"]
        for nct_id, data in trials_data.items():
            lines.append(f"### {nct_id}: {data['title']}")
            lines.append(f"- Phase: {data['phase']}")
            lines.append(f"- Status: {data['status']}")
            lines.append(f"- Enrollment: {data['enrollment']}")
            lines.append(f"- Start Date: {data['start_date']}")
            lines.append(
                f"- Conditions: {', '.join(data['conditions']) if data['conditions'] else 'N/A'}"
            )
            lines.append(
                f"- Interventions: {', '.join(data['interventions']) if data['interventions'] else 'N/A'}"
            )
            lines.append(
                f"- Primary Outcomes: {', '.join(data['primary_outcomes']) if data['primary_outcomes'] else 'N/A'}"
            )
            lines.append("")

        return "\n".join(lines)


class ExtractTrialFieldsTool:
    """Tool for extracting structured fields from a trial using LoRA model."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or create_stub_client()

    def execute(self, nct_id: str) -> dict:
        """Extract structured fields from trial."""
        with get_session() as session:
            from bioevidence.db.models import Trial

            trial = session.get(Trial, nct_id)
            if not trial:
                raise ValueError(f"Trial {nct_id} not found")

            # Build text from trial
            text_parts = [
                f"Title: {trial.title}",
                f"Phase: {trial.phase}",
                f"Status: {trial.overall_status}",
                f"Summary: {trial.brief_summary}",
                f"Description: {trial.detailed_description}",
                f"Eligibility: {trial.eligibility_text}",
            ]
            trial_text = "\n\n".join(filter(None, text_parts))

            # Use LLM to extract
            schema = {
                "primary_endpoint": "string",
                "endpoint_category": "string",
                "intervention_names": "array[string]",
                "intervention_class": "string",
                "randomized": "boolean",
                "blinding": "string",
                "population_age_group": "string",
                "key_inclusion": "array[string]",
                "key_exclusion": "array[string]",
            }

            prompt = f"""Extract structured fields from this clinical trial text:

{trial_text}

Return JSON with these fields:
{json.dumps(schema, indent=2)}"""

            response = self.llm.complete(prompt)
            try:
                extracted = json.loads(response.text)
                return {"nct_id": nct_id, "extracted": extracted}
            except json.JSONDecodeError:
                return {"nct_id": nct_id, "extracted": {}, "error": "Failed to parse extraction"}


class AgentTools:
    """Collection of all agent tools."""

    def __init__(self, system: str = "A5", llm_client: LLMClient | None = None):
        self.search = SearchEvidenceTool(system=system, llm_client=llm_client)
        self.sql = SQLQueryTool()
        self.compare = CompareTrialsTool(llm_client=llm_client)
        self.extract = ExtractTrialFieldsTool(llm_client=llm_client)
