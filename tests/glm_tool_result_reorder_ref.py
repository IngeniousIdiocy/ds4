#!/usr/bin/env python3
"""Render the GLM tool-result reordering test cases through the reference
Jinja chat template and print each expected prompt as a C string literal.

The literals are pasted into test_render_glm_reorders_tool_results() in
ds4_server.c; re-run this after a template update and diff the output.

Usage: python3 tests/glm_tool_result_reorder_ref.py [chat_template.jinja]

The template is the chat_template.jinja shipped in the official
zai-org/GLM-5.3-Flash snapshot (the same file the GGUF converter reads); the
path can also be given as GLM_CHAT_TEMPLATE in the environment.

Needs the jinja2 module (no pip install is attempted).  The environment is
built without trim_blocks/lstrip_blocks, which is the whitespace convention
the C port follows (see test_render_glm_preserves_reasoning_with_tools).

Message shapes: OpenAI-style cases use one role=tool message per response
with tool_call_id.  The Anthropic-style case (one user message carrying
several tool_result blocks) is modelled the way the template understands it,
as one role=tool message whose content is a list of {tool_call_id, output}
entries, since that is what ds4's Anthropic parser folds the blocks into.
"""
import json
import os
import sys

try:
    import jinja2
    from jinja2.ext import loopcontrols
except ImportError:
    sys.exit("jinja2 module not available")

path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GLM_CHAT_TEMPLATE")
if not path:
    sys.exit("usage: glm_tool_result_reorder_ref.py chat_template.jinja "
             "(or set GLM_CHAT_TEMPLATE)")
with open(path) as fh:
    src = fh.read()
env = jinja2.Environment(extensions=[loopcontrols], keep_trailing_newline=True)
env.policies["json.dumps_kwargs"] = {"ensure_ascii": False, "sort_keys": False}
tmpl = env.from_string(src)


def call(cid, cmd):
    return {"id": cid, "type": "function",
            "function": {"name": "bash", "arguments": {"command": cmd}}}


def tool(cid, text):
    return {"role": "tool", "tool_call_id": cid, "content": text}


def outputs(*pairs):
    return {"role": "tool",
            "content": [{"tool_call_id": cid, "output": text} if cid
                        else {"output": text} for cid, text in pairs]}


user = {"role": "user", "content": "run both"}
two = {"role": "assistant", "content": "",
       "tool_calls": [call("call_a", "pwd"), call("call_b", "ls")]}
three = {"role": "assistant", "content": "",
         "tool_calls": [call("call_a", "pwd"), call("call_b", "ls"),
                        call("call_c", "id")]}
dup = {"role": "assistant", "content": "",
       "tool_calls": [call("call_a", "pwd"), call("call_a", "ls")]}

cases = [
    ("in_order", [user, two, tool("call_a", "/tmp"), tool("call_b", "a b")]),
    ("out_of_order", [user, two, tool("call_b", "a b"), tool("call_a", "/tmp")]),
    ("missing_response", [user, three, tool("call_c", "uid=0"),
                          tool("call_a", "/tmp")]),
    ("duplicate_call_ids", [user, dup, tool("call_a", "a b"),
                            tool("call_a", "/tmp")]),
    ("duplicate_response_ids", [user, two, tool("call_b", "a b"),
                                tool("call_b", "again"), tool("call_a", "/tmp")]),
    ("anthropic_blocks", [user, two, outputs(("call_b", "a b"),
                                              ("call_a", "/tmp"))]),
    ("openai_messages", [user, two, tool("call_b", "a b"), tool("call_a", "/tmp")]),
    ("unmatched_response", [user, two, tool("call_b", "a b"),
                            tool("call_zzz", "stray"), tool("call_a", "/tmp")]),
    ("anthropic_dup_ids", [user, two, outputs(("call_b", "a b"),
                                               ("call_b", "/tmp"))]),
    ("anthropic_missing_id", [user, two, outputs(("call_b", "a b"),
                                                  (None, "/tmp"))]),
]

for name, messages in cases:
    text = tmpl.render(messages=messages, tools=None, add_generation_prompt=True,
                       reasoning_effort="high")
    print(f"/* {name} */")
    print(json.dumps(text, ensure_ascii=False))
