// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Target-specific replay policy for OpenAI Responses reasoning items.

use serde::Deserialize;
use serde_json::Value;

/// Controls which Responses reasoning items are replayed to an upstream.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesReasoningPolicy {
    /// Preserve provider-encrypted reasoning but remove plaintext reasoning.
    ///
    /// This is the safe default for strict hosted Responses providers.
    #[default]
    PreserveEncrypted,
    /// Drop all reasoning items while preserving messages and tool-call history.
    ///
    /// Use this for local Responses-compatible servers that cannot consume
    /// another provider's encrypted reasoning representation.
    Drop,
}

impl ResponsesReasoningPolicy {
    /// Normalizes a Responses request body for this replay policy.
    pub(crate) fn normalize(self, body: &mut Value) {
        let Some(Value::Array(input)) = body.get_mut("input") else {
            return;
        };
        input.retain_mut(|item| self.normalize_item(item));
    }

    fn normalize_item(self, item: &mut Value) -> bool {
        let Some(object) = item.as_object_mut() else {
            return true;
        };

        // ⛔ A Responses item that reaches an upstream without a `type` is rejected as
        // "Cannot determine type of 'item'" — measured against llama.cpp 2026-09-16.
        // A `{role, content}` item unambiguously IS a message, so name it rather than
        // forward an unroutable shape; dropping it would delete conversation content.
        //
        // ⚠ This is reached when one provider's replayed output is sent to another,
        // which is exactly what a tier switch does: a hosted turn's assistant items are
        // replayed to a local model on the next turn and the whole request 400s.
        // Measured asymmetry: a typeless item with `role: "user"` round-trips fine and
        // only the assistant side loses its `type`, so this normalises the case that is
        // actually broken rather than every item that omits the field.
        if !object.contains_key("type") && object.contains_key("role") {
            object.insert("type".to_string(), Value::from("message"));
        }

        if object.get("type").and_then(Value::as_str) != Some("reasoning") {
            return true;
        }

        let signed = matches!(
            object.get("encrypted_content").and_then(Value::as_str),
            Some(encrypted_content) if !encrypted_content.is_empty()
        );
        if self == Self::PreserveEncrypted && signed {
            object.insert("content".to_string(), Value::Array(Vec::new()));
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn mixed_history() -> Value {
        json!({
            "input": [
                {"type": "message", "role": "user", "content": []},
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "plaintext"}],
                    "encrypted_content": ""
                },
                {"type": "function_call", "call_id": "call_1"},
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "must be removed"}],
                    "encrypted_content": "encrypted"
                }
            ]
        })
    }

    #[test]
    fn preserve_encrypted_drops_unsigned_and_clears_plaintext() {
        let mut body = mixed_history();
        ResponsesReasoningPolicy::PreserveEncrypted.normalize(&mut body);

        let input = body["input"].as_array().expect("input array");
        let reasoning: Vec<&Value> = input
            .iter()
            .filter(|item| item["type"] == "reasoning")
            .collect();
        assert_eq!(reasoning.len(), 1);
        assert_eq!(reasoning[0]["encrypted_content"], "encrypted");
        assert_eq!(reasoning[0]["content"], json!([]));
        assert!(input.iter().any(|item| item["type"] == "function_call"));
        assert!(
            input
                .iter()
                .any(|item| item["type"] == "function_call_output")
        );
    }

    #[test]
    fn a_typeless_assistant_item_is_named_rather_than_forwarded() {
        // Reached on a tier switch: a hosted turn's assistant output replayed to a
        // local model. Without a `type` the upstream answers 400 and the turn dies.
        let mut body = json!({
            "input": [
                {"role": "assistant", "content": [{"type": "output_text", "text": "hi"}]}
            ]
        });
        ResponsesReasoningPolicy::Drop.normalize(&mut body);
        assert_eq!(body["input"][0]["type"], "message");
        assert_eq!(body["input"][0]["role"], "assistant");
        assert_eq!(
            body["input"][0]["content"][0]["text"], "hi",
            "content must survive being named"
        );
    }

    #[test]
    fn an_item_that_already_has_a_type_is_left_alone() {
        let mut body = json!({
            "input": [{"type": "function_call", "call_id": "c1", "role": "assistant"}]
        });
        ResponsesReasoningPolicy::Drop.normalize(&mut body);
        assert_eq!(body["input"][0]["type"], "function_call");
    }

    #[test]
    fn drop_removes_all_reasoning_and_keeps_tool_history() {
        let mut body = mixed_history();
        ResponsesReasoningPolicy::Drop.normalize(&mut body);

        let input = body["input"].as_array().expect("input array");
        assert!(input.iter().all(|item| item["type"] != "reasoning"));
        assert!(input.iter().any(|item| item["type"] == "function_call"));
        assert!(
            input
                .iter()
                .any(|item| item["type"] == "function_call_output")
        );
    }
}
