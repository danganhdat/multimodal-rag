"""Regression tests for agent conversation continuation."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from backend.app.agent import react


@contextmanager
def _span(*args, **kwargs):
    yield {}


def test_followup_resets_final_and_returns_an_answer(monkeypatch):
    class FakeGraph:
        initial_state = None

        async def aget_state(self, config):
            return SimpleNamespace(values={"messages": [HumanMessage(content="Earlier question")]})

        async def astream(self, initial_state, config, stream_mode):
            self.initial_state = initial_state
            yield {"agent": {"messages": [AIMessage(content="Follow-up response")]}}

    graph = FakeGraph()
    monkeypatch.setattr(react, "build_graph", lambda *a, **kw: (graph, 22))
    monkeypatch.setattr(react, "span", _span)

    async def collect():
        return [event async for event in react.stream_agent(
            "Next question", None, None, None, thread_id="existing-thread"
        )]

    events = asyncio.run(collect())
    assert events[0] == {"type": "meta", "thread_id": "existing-thread"}
    assert graph.initial_state["final"] is None
    assert graph.initial_state["messages"][0].content == "Next question"
    assert events[-1]["type"] == "answer"
    assert events[-1]["answer"] == "Follow-up response"
