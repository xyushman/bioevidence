"""Tests for BioEvidence agent multi-step reasoning."""

from bioevidence.agent.agent import (
    AgentResponse,
    BioEvidenceAgent,
    ToolCall,
    ToolName,
    ToolResult,
    create_agent,
)


class TestBioEvidenceAgent:
    """Tests for BioEvidenceAgent."""

    def test_agent_creation(self):
        """Test agent creation."""
        agent = create_agent(system="A5", max_steps=6)
        assert isinstance(agent, BioEvidenceAgent)
        assert agent.system == "A5"
        assert agent.max_steps == 6

    def test_parse_decision_valid_json(self):
        """Test parsing valid JSON decision."""
        agent = create_agent()
        text = '{"reasoning": "Search for cancer", "tool": "search_evidence", "args": {"query": "cancer"}, "answer": null}'
        decision = agent._parse_decision(text)
        assert decision is not None
        assert decision["tool"] == "search_evidence"
        assert decision["args"]["query"] == "cancer"

    def test_parse_decision_with_answer(self):
        """Test parsing decision with final answer."""
        agent = create_agent()
        text = '{"reasoning": "Found answer", "tool": null, "args": null, "answer": "Cancer is treated with pembrolizumab."}'
        decision = agent._parse_decision(text)
        assert decision is not None
        assert decision["answer"] == "Cancer is treated with pembrolizumab."
        assert decision["tool"] is None

    def test_parse_decision_embedded_json(self):
        """Test parsing JSON embedded in text."""
        agent = create_agent()
        text = 'Here is my decision: {"reasoning": "Search", "tool": "search_evidence", "args": {"query": "cancer"}, "answer": null}'
        decision = agent._parse_decision(text)
        assert decision is not None
        assert decision["tool"] == "search_evidence"

    def test_parse_decision_invalid(self):
        """Test parsing invalid JSON."""
        agent = create_agent()
        text = "This is not JSON at all"
        decision = agent._parse_decision(text)
        assert decision is None

    def test_format_messages(self):
        """Test message formatting."""
        agent = create_agent()
        messages = [
            {"role": "system", "content": "You are an assistant"},
            {"role": "user", "content": "Hello"},
        ]
        formatted = agent._format_messages(messages)
        assert "<|SYSTEM|>" in formatted
        assert "You are an assistant" in formatted
        assert "<|USER|>" in formatted
        assert "Hello" in formatted
        assert "<|assistant|>" in formatted


class TestToolName:
    """Tests for ToolName enum."""

    def test_tool_names(self):
        assert ToolName.SEARCH.value == "search_evidence"
        assert ToolName.SQL.value == "sql_query"
        assert ToolName.COMPARE.value == "compare_trials"
        assert ToolName.EXTRACT.value == "extract_trial_fields"


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        call = ToolCall(
            name=ToolName.SEARCH,
            args={"query": "cancer", "k": 5},
            reasoning="Need to find cancer trials",
        )
        assert call.name == ToolName.SEARCH
        assert call.args["query"] == "cancer"
        assert call.reasoning == "Need to find cancer trials"


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_tool_result_creation(self):
        result = ToolResult(
            name=ToolName.SEARCH,
            output="Found 5 trials",
            citations=[1, 2, 3],
        )
        assert result.name == ToolName.SEARCH
        assert result.output == "Found 5 trials"
        assert result.citations == [1, 2, 3]
        assert result.sql is None
        assert result.error is None

    def test_tool_result_with_error(self):
        result = ToolResult(
            name=ToolName.SQL,
            output="",
            error="Syntax error",
        )
        assert result.error == "Syntax error"


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""

    def test_agent_response_creation(self):
        response = AgentResponse(
            answer="Cancer is treated with pembrolizumab.",
            citations=[1, 2],
            tools_used=["search_evidence"],
            sql_queries=[],
            confident=True,
            steps=2,
        )
        assert response.answer == "Cancer is treated with pembrolizumab."
        assert response.citations == [1, 2]
        assert response.tools_used == ["search_evidence"]
        assert response.confident is True
        assert response.steps == 2


class TestCreateAgent:
    """Tests for create_agent factory."""

    def test_create_agent_defaults(self):
        agent = create_agent()
        assert agent.system == "A5"
        assert agent.max_steps == 6

    def test_create_agent_custom(self):
        agent = create_agent(system="A2", max_steps=10)
        assert agent.system == "A2"
        assert agent.max_steps == 10
