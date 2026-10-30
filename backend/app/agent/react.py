"""LangGraph ReAct agent for VQA.

StateGraph: agent → ToolNode → guard → END.
Supports both agentic mode (with search tool) and manual /ask mode (without).
"""

import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from ..config import settings
from ..llm import get_chat
from ..observe import span

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_KF_ID_PATTERN = re.compile(r"^L\d+_V\d+/\d{3,}$")
_checkpointer = MemorySaver()


def _load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    looked_kf_ids: list[str]
    looked_order: list[str]
    looked_texts: list[str]
    looked_image_paths: list[str]
    candidates: list[str]
    steps: int
    final: dict | None
    # Evidence memory
    active_video_id: str | None
    evidence_ids: list[str]
    observations: list[dict]
    # Budget counters
    search_calls: int
    look_calls: int


def build_graph(
    milvus,
    siglip2,
    bm25,
    mode: Literal["agentic", "manual"] = "agentic",
    flags: dict | None = None,
    trace_id: str = "",
    max_steps: int = 5,
    callbacks=None,
    context_log: dict | None = None,
):
    """Build the LangGraph agent graph.

    Args:
        mode: "agentic" includes search/search_events tools, "manual" only has look/neighbors/final_answer.
        flags: Feature flags for pipeline configuration.
        max_steps: max agent steps before forced termination.
    """
    flags = flags or {}
    from .tools import (
        make_search_tool,
        make_search_events_tool,
        make_read_ocr_tool,
        make_look_tool,
        make_crop_tool,
        make_neighbors_tool,
        final_answer,
    )

    tools = []
    if mode == "agentic":
        tools.append(make_search_tool(milvus, siglip2, bm25, flags=flags, trace_id=trace_id))
        tools.append(make_search_events_tool(milvus, siglip2, bm25, flags=flags, trace_id=trace_id))
    tools.append(make_read_ocr_tool(milvus, trace_id))
    tools.append(make_look_tool(milvus, trace_id, callbacks=callbacks))
    tools.append(make_crop_tool(trace_id, callbacks=callbacks))
    tools.append(make_neighbors_tool(milvus, trace_id))
    tools.append(final_answer)

    system_prompt = _load_prompt("react_v1.txt")
    chat = get_chat("text", temperature=0, max_tokens=2000,
                    extra_body={"reasoning": {"max_tokens": 512}}, timeout=90, callbacks=callbacks)
    model = chat.bind_tools(tools)

    tool_node = ToolNode(tools, handle_tool_errors=True)

    def agent_node(state: AgentState) -> dict:
        msgs = state["messages"]
        # Bound long conversations by approximate tokens, not message count.
        trimmed = trim_messages(
            msgs,
            max_tokens=flags.get("agent", {}).get("max_history_tokens", 4000),
            token_counter=lambda messages: sum(
                max(1, len(str(message.content)) // 4)
                + len(json.dumps(getattr(message, "tool_calls", []), default=str)) // 4
                + 8
                for message in messages
            ),
            strategy="last",
            start_on="human",
            include_system=True,
        )
        with span(trace_id, "agent.step", step=state["steps"]):
            response = model.invoke(trimmed)
        return {
            "messages": [response],
            "steps": state["steps"] + 1,
        }

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return END

        if last.tool_calls:
            # Check if it's calling final_answer
            for tc in last.tool_calls:
                if tc["name"] == "final_answer":
                    return "guard"
            return "tools"

        return END

    def guard_node(state: AgentState) -> dict:
        """Validate final_answer: check evidence kf_ids were looked at."""
        last_ai = state["messages"][-1]
        if not isinstance(last_ai, AIMessage) or not last_ai.tool_calls:
            return {}

        for tc in last_ai.tool_calls:
            if tc["name"] != "final_answer":
                continue
            args = tc["args"]
            status = args.get("status", "answered")
            raw_evidence = args.get("evidence", [])
            seen_ev = set()
            evidence = []
            for kf_id in raw_evidence:
                if kf_id not in seen_ev:
                    seen_ev.add(kf_id)
                    evidence.append(kf_id)
            answer = args.get("answer", "")

            guard_changed = False

            if status == "answered" and not evidence:
                guard_changed = True
                status = "uncertain"

            if status == "answered" and evidence:
                looked = set(state.get("looked_kf_ids", []))
                for kf_id in evidence:
                    if not _KF_ID_PATTERN.match(kf_id):
                        guard_changed = True
                        break
                    if kf_id not in looked:
                        guard_changed = True
                        break

                if guard_changed:
                    with span(trace_id, "agent.guard", guard_changed=True):
                        status = "uncertain"

            final = {
                "status": status,
                "answer": answer,
                "evidence": evidence,
                "guard_changed": guard_changed,
            }

            tool_msg = ToolMessage(
                content=json.dumps(final),
                tool_call_id=tc["id"],
            )
            return {"messages": [tool_msg], "final": final}

        return {}

    def update_looked(state: AgentState) -> dict:
        """After tool execution, track which kf_ids were looked at."""
        looked = set(state.get("looked_kf_ids", []))
        order = list(state.get("looked_order", []))
        paths = list(state.get("looked_image_paths", []))
        successful_calls = {msg.tool_call_id: msg for msg in state["messages"]
                            if isinstance(msg, ToolMessage) and msg.name in ("look", "crop")
                            and not str(msg.content).startswith("Error:")}
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] in ("look", "crop") and tc["id"] in successful_calls:
                        try:
                            observation = json.loads(successful_calls[tc["id"]].content)
                        except (ValueError, TypeError):
                            continue
                        kf_ids = observation.get("frame_ids", [])
                        looked.update(kf_ids)
                        for kf_id in kf_ids:
                            if kf_id not in order:
                                order.append(kf_id)
                        for path in observation.get("image_paths", []):
                            if path not in paths:
                                paths.append(path)
                break
        return {"looked_kf_ids": list(looked), "looked_order": order, "looked_image_paths": paths}

    async def tools_then_update(state: AgentState) -> dict:
        """Run tools, then update looked_kf_ids and budget counters."""
        # Budget check before executing
        agent_cfg = settings.to_dict().get("agent", {}) if hasattr(settings, "to_dict") else {}
        max_search = flags.get("agent", {}).get("max_search_calls", agent_cfg.get("max_search_calls", 3))
        max_look = flags.get("agent", {}).get("max_look_calls", agent_cfg.get("max_look_calls", 5))
        search_count = state.get("search_calls", 0)
        look_count = state.get("look_calls", 0)

        # Count tool calls in the last AI message
        last_ai = state["messages"][-1]
        new_search = 0
        new_look = 0
        if isinstance(last_ai, AIMessage) and last_ai.tool_calls:
            for tc in last_ai.tool_calls:
                if tc["name"] in ("search", "search_events"):
                    new_search += 1
                elif tc["name"] in ("look", "crop"):
                    new_look += 1

        if search_count + new_search > max_search or look_count + new_look > max_look:
            budget_msg = ToolMessage(
                content=json.dumps({"error": "budget_exceeded",
                                     "search_calls": search_count, "max_search": max_search,
                                     "look_calls": look_count, "max_look": max_look}),
                tool_call_id=last_ai.tool_calls[0]["id"] if last_ai.tool_calls else "",
            )
            return {
                "messages": [budget_msg],
                "search_calls": search_count,
                "look_calls": look_count,
            }

        tool_result = await tool_node.ainvoke(state)
        update = update_looked({**state, "messages": state["messages"] + tool_result["messages"]})

        # Update evidence memory
        evidence_ids = list(state.get("evidence_ids", []))
        observations = list(state.get("observations", []))
        active_video = state.get("active_video_id")

        for msg in tool_result["messages"]:
            if isinstance(msg, ToolMessage) and msg.name in ("look", "crop"):
                try:
                    obs = json.loads(msg.content)
                    for kf_id in obs.get("frame_ids", []):
                        if kf_id not in evidence_ids:
                            evidence_ids.append(kf_id)
                        vid = kf_id.split("/")[0] if "/" in kf_id else ""
                        if vid:
                            active_video = vid
                    observations.append({
                        "kf_ids": obs.get("frame_ids", []),
                        "observation": obs.get("observation", "")[:500],
                    })
                except (ValueError, TypeError):
                    pass

        context_texts = state.get("looked_texts", []) + [str(msg.content) for msg in tool_result["messages"]
                        if isinstance(msg, ToolMessage) and msg.name in ("look", "crop")
                        and not str(msg.content).startswith("Error:")]
        if context_log is not None:
            context_log.update({"context_frame_ids": update.get("looked_order", []),
                                "context_image_paths": update.get("looked_image_paths", []),
                                "context_texts": context_texts})

        return {
            "messages": tool_result["messages"],
            "looked_kf_ids": update.get("looked_kf_ids", state.get("looked_kf_ids", [])),
            "looked_order": update.get("looked_order", state.get("looked_order", [])),
            "looked_image_paths": update.get("looked_image_paths", state.get("looked_image_paths", [])),
            "looked_texts": context_texts,
            "active_video_id": active_video,
            "evidence_ids": evidence_ids,
            "observations": observations,
            "search_calls": search_count + new_search,
            "look_calls": look_count + new_look,
        }

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_then_update)
    graph.add_node("guard", guard_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "guard": "guard",
        END: END,
    })
    graph.add_edge("tools", "agent")
    graph.add_edge("guard", END)

    recursion_limit = 2 * max_steps + 2
    return graph.compile(checkpointer=_checkpointer), recursion_limit


async def run_agent(
    question: str,
    milvus,
    siglip2,
    bm25,
    mode: Literal["agentic", "manual"] = "agentic",
    candidates: list[str] | None = None,
    trace_id: str = "",
    max_steps: int = 5,
    timeout_s: int = 60,
    callbacks=None,
) -> dict:
    """Run the agent and return the final answer.

    Args:
        question: User's question.
        candidates: For manual mode, the kf_ids the user selected.
        trace_id: For observability.
        max_steps: Max agent iterations.
        timeout_s: Timeout in seconds.

    Returns:
        {"status", "answer", "evidence", "steps", "trace_id"}
    """
    import asyncio

    context_log = {"context_frame_ids": [], "context_image_paths": [], "context_texts": []}
    graph, recursion_limit = build_graph(
        milvus, siglip2, bm25,
        mode=mode, trace_id=trace_id, max_steps=max_steps,
        callbacks=callbacks,
        context_log=context_log,
    )

    system_prompt = _load_prompt("react_v1.txt")
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    if candidates:
        candidates_str = ", ".join(candidates)
        messages.append(HumanMessage(
            content=f"{'The user has selected' if mode == 'manual' else 'Search has retrieved'} these keyframes as candidates: {candidates_str}\n\n"
                    f"Question: {question}"
        ))
    else:
        messages.append(HumanMessage(content=question))

    import uuid as _uuid
    thread_id = str(_uuid.uuid4())
    initial_state: AgentState = {
        "messages": messages,
        "looked_kf_ids": [],
        "looked_order": [],
        "looked_texts": [],
        "looked_image_paths": [],
        "candidates": candidates or [],
        "steps": 0,
        "final": None,
        "active_video_id": None,
        "evidence_ids": [],
        "observations": [],
        "search_calls": 0,
        "look_calls": 0,
    }

    from ..llm import LLM_CALLBACKS
    callback_token = LLM_CALLBACKS.set(callbacks)
    config = {"recursion_limit": recursion_limit, "configurable": {"thread_id": thread_id}}
    with span(trace_id, "agent.run", mode=mode, question=question[:200]) as attrs:
        try:
            result = await asyncio.wait_for(
                graph.ainvoke(initial_state, config),
                timeout=timeout_s,
            )
            final = result.get("final")
            if final:
                attrs["status"] = final.get("status", "unknown")
                attrs["steps"] = result.get("steps", 0)
                attrs["search_calls"] = result.get("search_calls", 0)
                attrs["look_calls"] = result.get("look_calls", 0)
                return {
                    **final,
                    "steps": result.get("steps", 0),
                    "trace_id": trace_id,
                    "context_frame_ids": result.get("looked_order", []),
                    "context_texts": result.get("looked_texts", []),
                    "context_image_paths": result.get("looked_image_paths", []),
                    "active_video_id": result.get("active_video_id"),
                    "evidence_ids": result.get("evidence_ids", []),
                    "observations": result.get("observations", []),
                }

            attrs["status"] = "no_final_answer"
            return {
                "status": "uncertain",
                "answer": _extract_last_text(result.get("messages", [])),
                "evidence": [],
                "steps": result.get("steps", 0),
                "trace_id": trace_id,
                "context_frame_ids": result.get("looked_order", []),
                "context_texts": result.get("looked_texts", []),
                "context_image_paths": result.get("looked_image_paths", []),
            }

        except asyncio.TimeoutError:
            attrs["status"] = "timeout"
            return {
                "status": "uncertain",
                "answer": "Timeout: could not complete analysis in time.",
                "evidence": [],
                "steps": max_steps,
                "trace_id": trace_id,
                "timeout": True,
                **context_log,
            }
        except GraphRecursionError:
            attrs["status"] = "max_steps"
            return {
                "status": "uncertain",
                "answer": "Reached maximum steps without a definitive answer.",
                "evidence": [],
                "steps": max_steps,
                "trace_id": trace_id,
                **context_log,
            }
        finally:
            LLM_CALLBACKS.reset(callback_token)


async def stream_agent(
    question: str,
    milvus,
    siglip2,
    bm25,
    trace_id: str = "",
    max_steps: int = 10,
    timeout_s: int = 60,
    thread_id: str | None = None,
    flags: dict | None = None,
):
    """Stream agent events for SSE. Yields dicts with type + data."""
    import asyncio
    import uuid as _uuid

    graph, recursion_limit = build_graph(
        milvus, siglip2, bm25,
        mode="agentic", flags=flags, trace_id=trace_id, max_steps=max_steps,
    )

    if not thread_id:
        thread_id = str(_uuid.uuid4())

    config = {"recursion_limit": recursion_limit, "configurable": {"thread_id": thread_id}}

    # Detect follow-up: check if thread has saved state
    is_followup = False
    try:
        saved = await graph.aget_state(config)
        if saved and saved.values and saved.values.get("messages"):
            is_followup = True
    except Exception:
        pass

    if is_followup:
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "final": None,
            "steps": 0,
            "search_calls": 0,
            "look_calls": 0,
        }
    else:
        system_prompt = _load_prompt("react_v1.txt")
        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ],
            "looked_kf_ids": [],
            "looked_order": [],
            "looked_texts": [],
            "looked_image_paths": [],
            "candidates": [],
            "steps": 0,
            "final": None,
            "active_video_id": None,
            "evidence_ids": [],
            "observations": [],
            "search_calls": 0,
            "look_calls": 0,
        }

    yield {"type": "meta", "thread_id": thread_id}

    answered = False
    last_text = ""
    with span(trace_id, "agent.stream", question=question[:200], thread_id=thread_id):
        try:
            async with asyncio.timeout(timeout_s):
                async for event in graph.astream(
                    initial_state,
                    config,
                    stream_mode="updates",
                ):
                    for node_name, update in event.items():
                        if node_name == "agent":
                            msgs = update.get("messages", [])
                            for msg in msgs:
                                if isinstance(msg, AIMessage):
                                    if msg.content:
                                        last_text = str(msg.content)
                                        yield {"type": "thought", "content": last_text}
                                    for tc in (msg.tool_calls or []):
                                        yield {"type": "action", "tool": tc["name"], "args": tc["args"]}
                        elif node_name == "tools":
                            for msg in update.get("messages", []):
                                if isinstance(msg, ToolMessage):
                                    yield {"type": "observation", "tool": getattr(msg, "name", ""),
                                           "content": msg.content[:2000]}
                        elif node_name == "guard":
                            final = update.get("final")
                            if final:
                                answered = True
                                yield {"type": "answer", **final}
            if not answered:
                yield {"type": "answer", "status": "uncertain",
                       "answer": last_text or "The agent stopped without a final answer.",
                       "evidence": []}

        except GraphRecursionError:
            yield {
                "type": "answer",
                "status": "uncertain",
                "answer": "Reached maximum steps without a definitive answer.",
                "evidence": [],
            }
        except asyncio.TimeoutError:
            yield {
                "type": "answer",
                "status": "uncertain",
                "answer": "Timeout",
                "evidence": [],
            }


def _extract_last_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content[:500]
    return ""
