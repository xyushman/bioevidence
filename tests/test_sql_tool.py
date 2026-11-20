"""Tests for SQL tool with guardrails."""

import pytest

from bioevidence.agent.tools import SQLQueryTool


class TestSQLQueryTool:
    """Tests for SQLQueryTool."""

    def test_valid_select(self):
        """Test valid SELECT query passes validation."""
        tool = SQLQueryTool()
        valid, error = tool._validate_sql("SELECT * FROM trials WHERE phase = 'Phase 3'")
        assert valid is True
        assert error == ""

    def test_valid_select_with_cte(self):
        """Test valid WITH (CTE) query passes validation."""
        tool = SQLQueryTool()
        sql = (
            "WITH filtered AS (SELECT * FROM trials WHERE phase = 'Phase 3') SELECT * FROM filtered"
        )
        valid, error = tool._validate_sql(sql)
        assert valid is True

    def test_invalid_insert(self):
        """Test INSERT query is rejected."""
        tool = SQLQueryTool()
        valid, error = tool._validate_sql("INSERT INTO trials VALUES ('NCT001', 'Test')")
        assert valid is False
        assert "Invalid statement type" in error

    def test_invalid_update(self):
        """Test UPDATE query is rejected."""
        tool = SQLQueryTool()
        valid, error = tool._validate_sql("UPDATE trials SET phase = 'Phase 3'")
        assert valid is False

    def test_invalid_delete(self):
        """Test DELETE query is rejected."""
        tool = SQLQueryTool()
        valid, error = tool._validate_sql("DELETE FROM trials")
        assert valid is False

    def test_invalid_drop(self):
        """Test DROP query is rejected."""
        tool = SQLQueryTool()
        valid, error = tool._validate_sql("DROP TABLE trials")
        assert valid is False

    def test_blocked_function_pg_read_file(self):
        """Test pg_read_file is blocked."""
        tool = SQLQueryTool()
        valid, error = tool._validate_sql("SELECT pg_read_file('/etc/passwd')")
        assert valid is False
        assert "Blocked function" in error

    def test_blocked_function_information_schema(self):
        """Test information_schema access is blocked."""
        tool = SQLQueryTool()
        valid, error = tool._validate_sql("SELECT * FROM information_schema.tables")
        assert valid is False
        assert "Blocked function" in error

    def test_blocked_function_copy(self):
        """Test COPY is blocked."""
        tool = SQLQueryTool()
        valid, error = tool._validate_sql("COPY trials TO '/tmp/trials.csv'")
        assert valid is False

    def test_disallowed_table(self):
        """Test access to a table outside the allowlist is rejected."""
        tool = SQLQueryTool()
        valid, error = tool._validate_sql("SELECT * FROM users")
        assert valid is False
        assert "disallowed" in error

    def test_add_limit(self):
        """Test LIMIT is added when missing."""
        tool = SQLQueryTool()
        sql = "SELECT * FROM trials WHERE phase = 'Phase 3'"
        result = tool._ensure_limit(sql)
        assert "LIMIT 200" in result

    def test_keep_existing_limit(self):
        """Test existing LIMIT is preserved."""
        tool = SQLQueryTool()
        sql = "SELECT * FROM trials LIMIT 10"
        result = tool._ensure_limit(sql)
        assert result == sql

    def test_add_timeouts(self):
        """Test timeouts are added."""
        tool = SQLQueryTool()
        sql = "SELECT * FROM trials"
        result = tool._add_timeouts(sql)
        assert "statement_timeout" in result
        assert "lock_timeout" in result

    def test_validate_tables_allowed(self):
        """Test that allowed tables pass validation."""
        tool = SQLQueryTool()
        import sqlglot

        parsed = sqlglot.parse("SELECT * FROM trials", dialect="postgres")
        assert tool._validate_tables(parsed[0]) is True

    def test_validate_statement_select(self):
        """Test SELECT statement is valid."""
        tool = SQLQueryTool()
        import sqlglot

        parsed = sqlglot.parse("SELECT * FROM trials", dialect="postgres")
        assert tool._validate_statement(parsed[0]) is True

    def test_validate_statement_with(self):
        """Test WITH statement is valid."""
        tool = SQLQueryTool()
        import sqlglot

        parsed = sqlglot.parse(
            "WITH t AS (SELECT * FROM trials) SELECT * FROM t", dialect="postgres"
        )
        assert tool._validate_statement(parsed[0]) is True

    def test_validate_statement_invalid(self):
        """Test INSERT statement is invalid."""
        tool = SQLQueryTool()
        import sqlglot

        parsed = sqlglot.parse("INSERT INTO trials VALUES (1)", dialect="postgres")
        assert tool._validate_statement(parsed[0]) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
