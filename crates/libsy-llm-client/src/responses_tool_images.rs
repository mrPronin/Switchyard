// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Target-specific placement policy for images returned by a TOOL on the Responses wire.
//!
//! Some Responses-compatible servers reject an `input_image` inside a
//! `function_call_output`. llama.cpp is one: it converts Responses into its own chat
//! pipeline, and chat forbids image content in a `tool` message, so the request fails with
//! `400 "Output of tool call should be 'Input text'"`. The same image in a *user* message
//! is accepted, so the payload is legal — only its placement is not.
//!
//! ⛔ WHY THIS LIVES ON THE RESPONSES WIRE instead of being solved by translating to Chat.
//! Switching the client to `openai_chat` also fixes it, because the Chat codec re-homes
//! non-text tool output — but that re-renders the whole conversation into a second format,
//! and a re-rendered prefix stops matching the upstream's prompt cache. Measured on
//! llama.cpp: 92–95% cached going straight through, against 0–70% once translated, which
//! turned a 136 s fixture into a 1800 s timeout producing 241 tokens. Rewriting in place
//! keeps one renderer and therefore keeps the cache.
//!
//! ⚠ The rewrite is still a DEVIATION and is not silent: the model sees the image as a
//! user turn rather than as its tool's result. That is the same trade the Chat codec makes;
//! this policy only avoids paying for it twice.

use serde::Deserialize;
use serde_json::{Value, json};

/// Controls where an image returned by a tool is placed for an upstream.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesToolImagePolicy {
    /// Send the tool result exactly as the caller built it.
    ///
    /// Correct for hosted Responses providers, which accept an image inside a
    /// `function_call_output`.
    #[default]
    Inline,
    /// Move the image out of the tool result into a following user message.
    ///
    /// For upstreams that accept images only in a user turn — llama.cpp among them.
    Rehome,
}

/// What replaces an image that was lifted out of a tool result, so the tool call still
/// has an answer and the model is told where the picture went.
const MOVED_NOTE: &str = "[image returned by this tool follows in the next user message]";

impl ResponsesToolImagePolicy {
    /// Normalizes a Responses request body for this placement policy.
    pub(crate) fn normalize(self, body: &mut Value) {
        if self == Self::Inline {
            return;
        }
        let Some(Value::Array(input)) = body.get_mut("input") else {
            return;
        };
        let mut rewritten: Vec<Value> = Vec::with_capacity(input.len());
        for mut item in input.drain(..) {
            let images = lift_images(&mut item);
            rewritten.push(item);
            if !images.is_empty() {
                rewritten.push(json!({
                    "type": "message",
                    "role": "user",
                    "content": images,
                }));
            }
        }
        *input = rewritten;
    }
}

/// Removes every image block from one `function_call_output`, returning what was removed.
///
/// Leaves the tool result with its text (or a note, if the image WAS the whole answer) so
/// the call is still answered — an empty tool result is its own upstream error.
fn lift_images(item: &mut Value) -> Vec<Value> {
    let Some(object) = item.as_object_mut() else {
        return Vec::new();
    };
    if object.get("type").and_then(Value::as_str) != Some("function_call_output") {
        return Vec::new();
    }
    let Some(Value::Array(blocks)) = object.get_mut("output") else {
        return Vec::new();
    };
    let mut images = Vec::new();
    let mut kept = Vec::new();
    for block in blocks.drain(..) {
        if block.get("type").and_then(Value::as_str) == Some("input_image") {
            images.push(block);
        } else {
            kept.push(block);
        }
    }
    if images.is_empty() {
        *blocks = kept;
        return Vec::new();
    }
    if kept.is_empty() {
        kept.push(json!({"type": "input_text", "text": MOVED_NOTE}));
    }
    *blocks = kept;
    images
}
