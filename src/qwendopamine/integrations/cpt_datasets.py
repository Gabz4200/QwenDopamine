"""Dataset formatters for the CPT notebook.

Extracted from ``notebooks/train-infini-dopamine.py`` so the notebook
stays small and the formatters are unit-testable. The notebook imports
from here with a fallback to inline definitions for old Kaggle wheels.
"""

from __future__ import annotations

import json
from typing import Any


def _flatten_messages(messages: Any) -> str:
    if isinstance(messages, str):
        stripped = messages.strip()
        if stripped.startswith(("[", "{")):
            messages = json.loads(stripped)
        else:
            return messages
    if not isinstance(messages, list):
        return str(messages)
    parts = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", msg.get("from", "user"))
            content = msg.get("content", msg.get("value", msg.get("text", "")))
            reasoning = msg.get("reasoning_content", "")
            if reasoning:
                parts.append(f"{role}: [thinking: {reasoning}]\n{content}")
            else:
                parts.append(f"{role}: {content}")
        else:
            parts.append(str(msg))
    return "\n".join(parts)


def format_smb(example: dict) -> dict:
    return {"text": example.get("text", "")}


def format_maze(example: dict) -> dict:
    return {"text": example.get("text", "")}


def format_sokoban(example: dict) -> dict:
    messages_raw = example.get("messages", "")
    text = _flatten_messages(messages_raw)
    task = example.get("task", "")
    seed = example.get("seed", "")
    env_id = example.get("env_id", "")
    header = " | ".join(
        x for x in [f"task={task}", f"seed={seed}", f"env_id={env_id}"] if x
    )
    if header:
        text = f"[{header}]\n{text}"
    return {"text": text}


def format_bytesized32(example: dict) -> dict:
    prompt = example.get("prompt", [])
    reward_model = example.get("reward_model", "")
    extra_info = example.get("extra_info", "")
    parts = []
    if isinstance(prompt, list):
        for p in prompt:
            if isinstance(p, dict):
                role = p.get("role", "user")
                content = p.get("content", "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(p))
    elif prompt:
        parts.append(str(prompt))
    if reward_model:
        parts.append(f"RewardModel: {reward_model}")
    if extra_info:
        parts.append(f"ExtraInfo: {extra_info}")
    return {"text": "\n".join(parts)}


def format_patronus(example: dict) -> dict:
    text = _flatten_messages(example.get("messages", []))
    return {"text": text}


def format_arc(example: dict) -> dict:
    task = example.get("task", "")
    status = example.get("status", "")
    win_levels = example.get("win_levels", "")
    level_scores = []
    for k in [f"level{i}" for i in range(10)]:
        if k in example and example[k] is not None:
            level_scores.append(f"{k}={example[k]}")
    scores_str = ", ".join(level_scores)
    text = f"ARC-AGI Task [{task}] Status={status} WinLevels={win_levels}\nLevel Scores: {scores_str}"
    return {"text": text}


def format_chess_laion(example: dict) -> dict:
    moves = example.get("Moves", [])
    termination = example.get("Termination", "")
    result = example.get("Result", "*")
    if isinstance(moves, list):
        movetext = " ".join(str(m) for m in moves)
    else:
        movetext = str(moves)
    text = f'[Event "?"]\n[Result "{result}"]\n\n{movetext} {termination}'
    return {"text": text}


def format_openthoughts(example: dict) -> dict:
    system = example.get("system", "")
    convs = example.get("conversations", [])
    parts = []
    if system:
        parts.append(f"system: {system}")
    for turn in convs:
        role = turn.get("from", turn.get("role", "unknown"))
        value = turn.get("value", turn.get("content", ""))
        parts.append(f"{role}: {value}")
    return {"text": "\n".join(parts)}


def format_alfworld(example: dict) -> dict:
    steps_raw = example.get("steps", "[]")
    steps: list[Any]
    if isinstance(steps_raw, str):
        stripped = steps_raw.strip()
        if stripped.startswith(("[", "{")):
            steps = json.loads(stripped)
        else:
            steps = []
    else:
        steps = steps_raw if isinstance(steps_raw, list) else []
    parts = []
    task = example.get("task", "")
    task_type = example.get("task_type", "")
    if task:
        parts.append(f"Task: {task}")
    if task_type:
        parts.append(f"Task Type: {task_type}")
    for step in steps:
        idx = step.get("idx", step.get("step", step.get("id", "")))
        obs = step.get("obs", step.get("observation", step.get("text", "")))
        action = step.get("action", step.get("act", ""))
        if obs:
            parts.append(f"Step {idx} Observation: {obs}")
        if action:
            parts.append(f"Step {idx} Action: {action}")
    return {"text": "\n".join(parts)}


def format_kimi_k3(example: dict) -> dict:
    text = _flatten_messages(example.get("messages", []))
    return {"text": text}


def format_cot_eval(example: dict) -> dict:
    parts = []
    passage = example.get("passage", "")
    if passage:
        parts.append(f"Passage: {passage}")
    question = example.get("question", "")
    if question:
        parts.append(f"Question: {question}")
    options = example.get("options", [])
    if options:
        opts = " | ".join(str(o) for o in options)
        parts.append(f"Options: {opts}")
    answer = example.get("answer", "")
    if answer:
        parts.append(f"Answer: {answer}")
    trace = example.get("reasoning_trace", "")
    if trace:
        parts.append(f"Reasoning: {trace}")
    return {"text": "\n".join(parts)}


def format_lichess(example: dict) -> dict:
    movetext = example.get("movetext", "")
    white = str(example.get("White") or "?")
    black = str(example.get("Black") or "?")
    result = example.get("Result", "*")
    opening = example.get("Opening", "")
    eco = example.get("ECO", "")
    event = example.get("Event", "")
    site = example.get("Site", "")
    date = example.get("UTCDate", "")
    text = f'[Event "{event}"]\n[Site "{site}"]\n[Date "{date}"]\n[White "{white}"]\n[Black "{black}"]\n[Result "{result}"]\n[ECO "{eco}"]\n[Opening "{opening}"]\n\n{movetext}'
    return {"text": text}


def format_toolace(example: dict) -> dict:
    system = example.get("system", "")
    convs = example.get("conversations", [])
    parts = []
    if system:
        parts.append(f"system: {system}")
    for turn in convs:
        role = turn.get("from", turn.get("role", "unknown"))
        value = turn.get("value", turn.get("content", ""))
        parts.append(f"{role}: {value}")
    return {"text": "\n".join(parts)}


def format_qwen3_distill(example: dict) -> dict:
    text = _flatten_messages(example.get("messages", []))
    domain = example.get("domain", "")
    category = example.get("category", "")
    source = example.get("source", "")
    meta = " | ".join(
        x for x in [f"domain={domain}", f"category={category}", f"source={source}"] if x
    )
    if meta:
        text = f"[{meta}]\n{text}"
    return {"text": text}


def format_fable5(example: dict) -> dict:
    text = _flatten_messages(example.get("messages", []))
    trace = example.get("trace", "")
    prompt = example.get("prompt", "")
    parts = []
    if prompt:
        parts.append(f"Prompt: {prompt}")
    parts.append(text)
    if trace:
        parts.append(f"Trace: {trace}")
    return {"text": "\n".join(parts)}


def format_wikitext(example: dict) -> dict:
    text = example.get("text", "")
    if not text or not text.strip():
        text = "[EMPTY_WIKITEXT_ROW]"
    return {"text": text}


def format_r0b0tlab(example: dict) -> dict:
    raw: list[Any] = []
    raw_value = example.get("messages_json", "[]")
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped.startswith(("[", "{")):
            raw = json.loads(stripped)
        else:
            raw = []
    else:
        raw = raw_value if isinstance(raw_value, list) else []
    text = _flatten_messages(raw)
    task_type = example.get("task_type", "")
    source = example.get("source", "")
    domain = example.get("domain", "")
    meta = " | ".join(
        x
        for x in [f"task_type={task_type}", f"source={source}", f"domain={domain}"]
        if x
    )
    if meta:
        text = f"[{meta}]\n{text}"
    return {"text": text}


def format_arc2(example: dict) -> dict:
    """zhmz90/arc-agi-2: train/test grids with filename."""
    filename = example.get("filename", "")
    parts = [f"ARC-AGI-2 [{filename}]"]
    train = example.get("train", [])
    test = example.get("test", [])
    if train:
        parts.append(f"Train examples: {len(train)}")
        for i, ex in enumerate(train[:3]):
            inp = ex.get("input", ex) if isinstance(ex, dict) else ex
            out = ex.get("output", "") if isinstance(ex, dict) else ""
            parts.append(f"  Train {i} input: {inp} -> output: {out}")
    if test:
        parts.append(f"Test examples: {len(test)}")
        for i, ex in enumerate(test[:2]):
            inp = ex.get("input", ex) if isinstance(ex, dict) else ex
            out = ex.get("output", "") if isinstance(ex, dict) else ""
            parts.append(f"  Test {i} input: {inp} -> output: {out}")
    return {"text": "\n".join(parts)}


def format_chain_of_draft(example: dict) -> dict:
    """dvilasuero/chain-of-draft-r1: question + CoD vs standard."""
    question = example.get("question", "")
    answer = example.get("answer", "")
    cod = example.get("cod", "")
    standard = example.get("standard", "")
    parts = []
    if question:
        parts.append(f"Question: {question}")
    reasoning = cod.strip() if cod and cod.strip() else standard
    if reasoning:
        parts.append(f"Reasoning: {reasoning}")
    if answer:
        parts.append(f"Answer: {answer}")
    return {"text": "\n".join(parts)}


def format_nemotron(example: dict) -> dict:
    """nvidia/Nemotron-SFT-ARC-AGI-v1: messages with system+user ARC puzzle."""
    text = _flatten_messages(example.get("messages", []))
    tools = example.get("tools", [])
    if tools:
        tool_names = ", ".join(t.get("name", "?") for t in tools if isinstance(t, dict))
        text = f"[Tools: {tool_names}]\n{text}"
    meta = example.get("metadata", "")
    if meta:
        text = f"{text}\n[metadata: {meta}]"
    return {"text": text}


def format_ara_agent(example: dict) -> dict:
    """AgentNativeResearchLab ARC-AGI-3 agent trajectories (episodes + ara stats)."""
    if "frame" in example:
        turn = example.get("turn", "")
        action = example.get("action", "")
        state = example.get("state", "")
        levels = example.get("levels_completed", "")
        frame = str(example.get("frame", ""))[:1200]
        return {
            "text": f"ARC3 Episode turn={turn} action={action} state={state} levels={levels}\nFrame:\n{frame}"
        }
    if "ts" in example:
        turn = example.get("turn", "")
        trace_nodes = example.get("trace_nodes", "")
        ara_bytes = example.get("ara_bytes", "")
        claims = example.get("claims", "")
        return {
            "text": f"ARC3 Trace ts={example.get('ts', '')} turn={turn} nodes={trace_nodes} bytes={ara_bytes} claims={claims}"
        }
    return {
        "text": " ".join(
            str(v)[:500] for v in example.values() if isinstance(v, (str, int, float))
        )
    }


DATASET_FORMATTERS = {
    "DylanRiden/smb-worldmodel-data": format_smb,
    "Kalso42/WorldModelForMaze": format_maze,
    "ultrastar111/sokoban_easy_v8_cot_chunk_kinf_world_model_20260707_perseg": format_sokoban,
    "thuml/bytesized32-world-model-cot": format_bytesized32,
    "PatronusAI/world_model_corpus": format_patronus,
    "schema-harness/arc-agi-3-schema-traces": format_arc,
    "laion/strategic_game_chess": format_chess_laion,
    "ryanmarten/OpenThoughts-1k-sample": format_openthoughts,
    "Decix/ReBel-ALFWorld-SFT-Trajectories": format_alfworld,
    "greghavens/kimi-k3-coding-and-debugging-traces": format_kimi_k3,
    "cot-leaderboard/cot-eval-traces-2.0": format_cot_eval,
    "Lichess/standard-chess-games": format_lichess,
    "lockon/ToolACE": format_toolace,
    "faunix/Qwen3.8-27B-Distillation-40K": format_qwen3_distill,
    "Glint-Research/Fable-5-traces": format_fable5,
    "Salesforce/wikitext": format_wikitext,
    "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation": format_r0b0tlab,
    "zhmz90/arc-agi-2": format_arc2,
    "dvilasuero/chain-of-draft-r1": format_chain_of_draft,
    "nvidia/Nemotron-SFT-ARC-AGI-v1": format_nemotron,
    "AgentNativeResearchLab/arc-agi3-codex-gpt5.5-s5i5": format_ara_agent,
    "AgentNativeResearchLab/arc-agi3-kimi-k2.7-g50t": format_ara_agent,
    "AgentNativeResearchLab/arc-agi3-codex-gpt5.6sol-r11l": format_ara_agent,
    "AgentNativeResearchLab/arc-agi3-codex-gpt5.5-r11l": format_ara_agent,
}


def format_example(example: dict, dataset_name: str) -> dict:
    formatter = DATASET_FORMATTERS.get(dataset_name)
    if formatter is not None:
        return formatter(example)
    for col in ["text", "content", "prompt", "problem", "solution"]:
        if example.get(col):
            return {"text": str(example[col])}
    text = " ".join(
        str(v)
        for v in example.values()
        if isinstance(v, (str, int, float)) and not str(v).startswith("_")
    )
    return {"text": text}


__all__ = ["DATASET_FORMATTERS", "format_example"]
