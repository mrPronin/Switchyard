// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Target-specific replay policy for OpenAI Responses CUSTOM (freeform) tool items.
//!
//! A custom tool takes a raw string instead of JSON arguments, and its call is reported as
//! `custom_tool_call` / `custom_tool_call_output`. Codex declares `apply_patch` this way
//! for the GPT-5 family, so ANY hosted turn that edits a file leaves a pair of these items
//! in the conversation for every later turn to replay.
//!
//! ⛔ Some Responses-compatible servers do not model those two item types and reject the
//! WHOLE request with `400 "Cannot determine type of 'item'"` — the same message they emit
//! for an item carrying no `type` at all. That collision is why this read for months as a
//! missing-field bug: see `responses_reasoning`, which fixes the genuinely typeless case
//! and cannot fix this one, because these items *have* a type and it is simply unknown.
//!
//! Measured against llama.cpp on 2026-09-17, one item added to a minimal request:
//!
//! | item in `input`                             | status |
//! |---------------------------------------------|--------|
//! | `message`                                   | 200    |
//! | `custom_tool_call`                          | 400    |
//! | `custom_tool_call_output`                   | 400    |
//! | the same call as `function_call` + `_output`| 200    |
//!
//! ⚠ This is reached on a TIER SWITCH: a hosted turn's `apply_patch` replayed to a local
//! model on the next turn kills the request before any token is generated.
//!
//! ⭐ `tools` IS DELIBERATELY LEFT ALONE. A `{"type": "custom"}` declaration is accepted by
//! the same server with 200 — only the replayed history is rejected — so the model still
//! answers in the shape the caller declared and nothing has to be translated back.
//!
//! ⭐ The rewrite is lossless. An output differs only in its `type`: both carry the same
//! `{call_id, output}`. A call differs only in how the payload is spelled — `input` is raw
//! text where `arguments` is a JSON string — so the text is encoded into a single-field
//! object and the `call_id` that pairs the two is untouched.

use serde::Deserialize;
use serde_json::{Value, json};

/// The field the freeform `input` is carried in once it becomes JSON `arguments`.
const ARGUMENT_FIELD: &str = "input";

/// Controls how Responses custom (freeform) tool items are replayed to an upstream.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesCustomToolPolicy {
    /// Replay custom tool items exactly as the caller built them.
    ///
    /// Correct for hosted Responses providers, which define these item types.
    #[default]
    Preserve,
    /// Rewrite custom tool items as their `function_call` equivalents.
    ///
    /// For upstreams that model only function tools — llama.cpp among them.
    Translate,
}

impl ResponsesCustomToolPolicy {
    /// Normalizes a Responses request body for this replay policy.
    pub(crate) fn normalize(self, body: &mut Value) {
        if self == Self::Preserve {
            return;
        }
        let Some(Value::Array(input)) = body.get_mut("input") else {
            return;
        };
        for item in input.iter_mut() {
            translate_item(item);
        }
    }
}

/// Rewrites one custom tool item in place; anything else is left exactly as it is.
fn translate_item(item: &mut Value) {
    let Some(object) = item.as_object_mut() else {
        return;
    };
    match object.get("type").and_then(Value::as_str) {
        Some("custom_tool_call") => {
            let raw = object.remove(ARGUMENT_FIELD);
            object.insert("type".to_string(), Value::from("function_call"));
            object.insert("arguments".to_string(), Value::from(freeform_arguments(raw)));
        }
        Some("custom_tool_call_output") => {
            object.insert("type".to_string(), Value::from("function_call_output"));
        }
        _ => {}
    }
}

/// Encodes a freeform tool `input` as the JSON string a `function_call` expects.
///
/// A missing or non-string `input` is still encoded rather than dropped: an argumentless
/// tool call is legal, and a malformed one is the caller's to see, not ours to swallow.
fn freeform_arguments(raw: Option<Value>) -> String {
    let text = match raw {
        Some(Value::String(text)) => text,
        Some(other) => other.to_string(),
        None => String::new(),
    };
    serde_json::to_string(&json!({ ARGUMENT_FIELD: text })).unwrap_or_else(|_| "{}".to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// The shape Codex replays after a hosted turn used `apply_patch`.
    fn hosted_apply_patch() -> Value {
        json!({
            "input": [
                {"type": "message", "role": "user", "content": []},
                {
                    "type": "custom_tool_call",
                    "id": "ctc_1",
                    "status": "completed",
                    "call_id": "call_1",
                    "name": "apply_patch",
                    "input": "*** Begin Patch\n*** Add File: DESIGN.md\n+hi\n*** End Patch"
                },
                {
                    "type": "custom_tool_call_output",
                    "id": "ctco_1",
                    "call_id": "call_1",
                    "output": "Success. Updated the following files:\nA DESIGN.md\n"
                }
            ]
        })
    }

    #[test]
    fn translate_rewrites_a_custom_call_as_a_function_call() {
        let mut body = hosted_apply_patch();
        ResponsesCustomToolPolicy::Translate.normalize(&mut body);

        let call = &body["input"][1];
        assert_eq!(call["type"], "function_call");
        assert_eq!(call["name"], "apply_patch");
        assert_eq!(call["call_id"], "call_1");
        assert!(call.get("input").is_none(), "raw input must not survive alongside arguments");

        let arguments: Value = serde_json::from_str(call["arguments"].as_str().expect("a string"))
            .expect("arguments must be a JSON string");
        assert_eq!(
            arguments["input"], "*** Begin Patch\n*** Add File: DESIGN.md\n+hi\n*** End Patch",
            "the patch text must survive being re-spelled"
        );
    }

    #[test]
    fn translate_rewrites_the_output_and_keeps_the_pairing() {
        let mut body = hosted_apply_patch();
        ResponsesCustomToolPolicy::Translate.normalize(&mut body);

        let output = &body["input"][2];
        assert_eq!(output["type"], "function_call_output");
        assert_eq!(output["call_id"], body["input"][1]["call_id"]);
        assert_eq!(
            output["output"], "Success. Updated the following files:\nA DESIGN.md\n",
            "an output differs only in its type"
        );
    }

    #[test]
    fn preserve_is_the_default_and_leaves_custom_items_alone() {
        let mut body = hosted_apply_patch();
        let before = body.clone();
        ResponsesCustomToolPolicy::default().normalize(&mut body);
        assert_eq!(ResponsesCustomToolPolicy::default(), ResponsesCustomToolPolicy::Preserve);
        assert_eq!(body, before);
    }

    #[test]
    fn translate_leaves_every_other_item_untouched() {
        let mut body = json!({
            "input": [
                {"type": "message", "role": "assistant", "content": []},
                {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
                {"type": "reasoning", "encrypted_content": "e"}
            ]
        });
        let before = body.clone();
        ResponsesCustomToolPolicy::Translate.normalize(&mut body);
        assert_eq!(body, before);
    }

    #[test]
    fn translate_leaves_the_tool_declaration_alone() {
        // Measured: a `custom` declaration is accepted with 200, so translating it would
        // change the answer's shape for no reason.
        let mut body = json!({
            "tools": [{"type": "custom", "name": "apply_patch"}],
            "input": [{"type": "custom_tool_call", "call_id": "c1", "name": "apply_patch",
                       "input": "patch"}]
        });
        ResponsesCustomToolPolicy::Translate.normalize(&mut body);
        assert_eq!(body["tools"][0]["type"], "custom");
        assert_eq!(body["input"][0]["type"], "function_call");
    }

    #[test]
    fn a_call_without_input_still_becomes_a_function_call() {
        let mut body = json!({
            "input": [{"type": "custom_tool_call", "call_id": "c1", "name": "noop"}]
        });
        ResponsesCustomToolPolicy::Translate.normalize(&mut body);
        assert_eq!(body["input"][0]["type"], "function_call");
        assert_eq!(body["input"][0]["arguments"], r#"{"input":""}"#);
    }

    #[test]
    fn a_body_without_input_is_not_a_panic() {
        let mut body = json!({"model": "m"});
        ResponsesCustomToolPolicy::Translate.normalize(&mut body);
        assert_eq!(body, json!({"model": "m"}));
    }
}
