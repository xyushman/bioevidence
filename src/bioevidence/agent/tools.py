"""Agent tools for BioEvidence."""

import json
from dataclasses import dataclass

import sqlglot
import structlog
from sqlglot import exp

from bioevidence.config import get_config
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
    """Tool for executing read-only SQL queries with guardrails."""

    ALLOWED_TABLES = frozenset(
        {
            "trials",
            "trial_conditions",
            "trial_interventions",
            "trial_outcomes",
            "pubmed_articles",
            "drug_labels",
            "chunks",
        }
    )
    BLOCKED_FUNCTIONS = frozenset(
        {
            "pg_",
            "information_schema",
            "copy",
            "into",
            "pg_read_file",
            "pg_write_file",
            "pg_ls_dir",
            "pg_stat_file",
            "pg_catalog",
            "pg_roles",
            "pg_user",
            "pg_authid",
        }
    )
    BLOCKED_STATEMENTS = frozenset(
        {
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "TRUNCATE",
            "GRANT",
            "REVOKE",
            "COPY",
            "EXPLAIN",
            "PREPARE",
            "EXECUTE",
            "DECLARE",
            "CURSOR",
            "BEGIN",
            "COMMIT",
            "ROLLBACK",
            "SAVEPOINT",
            "SET",
            "RESET",
            "SHOW",
            "LOAD",
            "CHECKPOINT",
            "VACUUM",
            "ANALYZE",
        }
    )
    MAX_LIMIT = 200
    STATEMENT_TIMEOUT_MS = 3000
    LOCK_TIMEOUT_MS = 1000

    def __init__(self):
        self.config = get_config()

    def _validate_sql(self, sql: str) -> tuple[bool, str]:
        """Validate SQL using sqlglot AST parsing."""
        try:
            # Parse SQL
            parsed = sqlglot.parse(sql, dialect="postgres")
            if not parsed:
                return False, "Failed to parse SQL"

            for statement in parsed:
                if not self._validate_statement(statement):
                    return False, "Invalid statement type"

            # Check for blocked functions in raw SQL (case-insensitive)
            sql_lower = sql.lower()
            for blocked in self.BLOCKED_FUNCTIONS:
                if blocked in sql_lower:
                    return False, f"Blocked function/keyword: {blocked}"

            # Check table references
            for statement in parsed:
                if not self._validate_tables(statement):
                    return False, "References to disallowed tables"
        except sqlglot.ParseError as e:
            return False, f"SQL parse error: {e}"
        except Exception as e:
            logger.exception("sql_validation_failed")
            return False, f"Validation error: {e}"
        else:
            return True, ""

    def _validate_statement(self, statement: exp.Expression) -> bool:
        """Validate statement type is SELECT only."""
        # Allow SELECT, WITH (CTE)
        return isinstance(statement, (exp.Select, exp.With))

    def _validate_tables(self, statement: exp.Expression) -> bool:
        """Validate all table references are in allowed list."""
        # Get CTE names to exclude from validation
        cte_names = set()
        for cte in statement.find_all(exp.CTE):
            cte_names.add(cte.alias.lower())

        for table in statement.find_all(exp.Table):
            table_name = table.name.lower()
            # Skip CTE references
            if table_name in cte_names:
                continue
            if table_name not in self.ALLOWED_TABLES:
                logger.warning("disallowed_table", table=table_name)
                return False
        return True

    def _add_timeouts(self, sql: str) -> str:
        """Add statement and lock timeouts to SQL."""
        # Remove trailing semicolon
        sql = sql.rstrip().rstrip(";")

        # Add timeouts via SET LOCAL
        timeout_sql = f"""
SET LOCAL statement_timeout = {self.STATEMENT_TIMEOUT_MS};
SET LOCAL lock_timeout = {self.LOCK_TIMEOUT_MS};
{sql};
"""
        return timeout_sql

    def _ensure_limit(self, sql: str) -> str:
        """Ensure LIMIT clause is present."""
        sql_lower = sql.lower()
        if "limit" not in sql_lower:
            return f"{sql.rstrip().rstrip(';')} LIMIT {self.MAX_LIMIT}"
        return sql

    def execute(self, sql: str) -> list[dict]:
        """Execute SQL query with guardrails."""
        # Validate
        valid, error = self._validate_sql(sql)
        if not valid:
            raise ValueError(f"SQL validation failed: {error}")

        # Add LIMIT if not present
        sql = self._ensure_limit(sql)

        # Add timeouts
        sql = self._add_timeouts(sql)

        try:
            with get_session() as session:
                # Set timeouts at session level
                session.execute(f"SET LOCAL statement_timeout = {self.STATEMENT_TIMEOUT_MS}")
                session.execute(f"SET LOCAL lock_timeout = {self.LOCK_TIMEOUT_MS}")

                result = session.execute(sql)
                columns = result.keys()
                rows = [dict(zip(columns, row, strict=False)) for row in result.fetchall()]
                return rows

        except Exception as e:
            logger.exception("sql_execution_failed")
            raise ValueError(f"SQL execution failed: {e}") from e


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
            except json.JSONDecodeError:
                return {"nct_id": nct_id, "extracted": {}, "error": "Failed to parse extraction"}
            else:
                return {"nct_id": nct_id, "extracted": extracted}


class AgentTools:
    """Collection of all agent tools."""

    def __init__(self, system: str = "A5", llm_client: LLMClient | None = None):
        self.search = SearchEvidenceTool(system=system, llm_client=llm_client)
        self.sql = SQLQueryTool()
        self.compare = CompareTrialsTool(llm_client=llm_client)
        self.extract = ExtractTrialFieldsTool(llm_client=llm_client)
