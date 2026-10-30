"""Agent contexts must reflect successful asynchronous image inspections."""

import asyncio
import json
from contextlib import contextmanager

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from backend.app.agent import react, tools


def test_agent_keeps_conversation_and_records_only_images_actually_inspected(monkeypatch):
    @contextmanager
    def fake_span(*args, **kwargs):
        yield {}

    @tool("look", args_schema=tools.LookArgs)
    async def fake_look(kf_ids, question):
        """Inspect one available image; the second requested image is missing."""
        return json.dumps({"frame_ids": ["L21_V001/001"], "image_paths": ["available.jpg"],
                           "observation": "A red car is visible."})

    class FakeChat:
        calls = 0

        def bind_tools(self, available):
            assert "crop" in {t.name for t in available}
            return self

        def invoke(self, messages):
            assert any(isinstance(m, HumanMessage) for m in messages)
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{"id": "look1", "name": "look", "args": {
                    "kf_ids": ["L21_V001/001", "L21_V001/002"], "question": "What is visible?"}}])
            assert any(isinstance(m, ToolMessage) for m in messages)
            return AIMessage(content="", tool_calls=[{"id": "final1", "name": "final_answer", "args": {
                "status": "answered", "answer": "A red car.", "evidence": ["L21_V001/001"]}}])

    monkeypatch.setattr(react, "span", fake_span)
    monkeypatch.setattr(react, "get_chat", lambda *a, **k: FakeChat())
    monkeypatch.setattr(tools, "make_look_tool", lambda *a, **k: fake_look)
    result = asyncio.run(react.run_agent("What is visible?", None, None, None, mode="manual",
                                         candidates=["L21_V001/001", "L21_V001/002"]))
    assert result["status"] == "answered"
    assert result["context_frame_ids"] == ["L21_V001/001"]
    assert result["context_image_paths"] == ["available.jpg"]
    assert "red car" in result["context_texts"][0]
