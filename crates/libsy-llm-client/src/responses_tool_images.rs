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

#[cfg(test)]
mod tests {
    use super::*;

    /// A tool result carrying `blocks`, in the shape the Responses wire uses.
    fn tool_output(blocks: Value) -> Value {
        json!({"type": "function_call_output", "call_id": "c1", "output": blocks})
    }

    fn image(url: &str) -> Value {
        json!({"type": "input_image", "image_url": url})
    }

    fn body(input: Value) -> Value {
        json!({"model": "m", "input": input})
    }

    /// ⛔ The default must be byte-for-byte inert. `Inline` is what every hosted
    /// Responses provider needs, so a policy that "helpfully" rewrote anything here
    /// would corrupt the majority case to serve the minority one.
    #[test]
    fn inline_changes_nothing_at_all() {
        let original = body(json!([
            tool_output(json!([image("data:image/png;base64,AAA")])),
            {"type": "message", "role": "user", "content": []},
        ]));
        let mut subject = original.clone();
        ResponsesToolImagePolicy::Inline.normalize(&mut subject);
        assert_eq!(subject, original);
        assert_eq!(
            ResponsesToolImagePolicy::default(),
            ResponsesToolImagePolicy::Inline,
            "the default must stay Inline — Rehome is a deviation, opted into per client"
        );
    }

    /// The whole point: the image leaves the tool result and arrives as the NEXT item,
    /// as a user message. ⚠ Position matters — the model has to see it after the call
    /// it answers, not at the end of the conversation.
    #[test]
    fn rehome_moves_the_image_into_the_immediately_following_user_message() {
        let mut subject = body(json!([
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "look"}]},
            tool_output(json!([image("data:image/png;base64,AAA")])),
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "after"}]},
        ]));
        ResponsesToolImagePolicy::Rehome.normalize(&mut subject);
        let input = subject["input"].as_array().expect("input stays an array");

        assert_eq!(input.len(), 4, "one item added, none lost: {input:#?}");
        assert_eq!(input[1]["type"], "function_call_output");
        assert_eq!(input[2]["role"], "user");
        assert_eq!(input[2]["content"][0]["type"], "input_image");
        assert_eq!(
            input[3]["content"][0]["text"], "after",
            "the conversation after the tool result must keep its place"
        );
    }

    /// ⛔ An empty tool result is its own upstream error, so lifting the only block out
    /// has to leave something behind. The note is also the model's only clue that the
    /// picture moved.
    #[test]
    fn an_image_only_result_keeps_a_note_rather_than_becoming_empty() {
        let mut subject = body(json!([tool_output(json!([image("u")]))]));
        ResponsesToolImagePolicy::Rehome.normalize(&mut subject);
        let out = &subject["input"][0]["output"];

        assert_eq!(out.as_array().map(Vec::len), Some(1), "{out:#?}");
        assert_eq!(out[0]["type"], "input_text");
        assert_eq!(out[0]["text"], MOVED_NOTE);
    }

    /// ...and when the tool DID say something, that text is the answer and no note is
    /// invented on top of it.
    #[test]
    fn accompanying_text_survives_and_gets_no_note() {
        let mut subject = body(json!([tool_output(json!([
            {"type": "input_text", "text": "1024x768"},
            image("u"),
        ]))]));
        ResponsesToolImagePolicy::Rehome.normalize(&mut subject);
        let out = &subject["input"][0]["output"];

        assert_eq!(out.as_array().map(Vec::len), Some(1), "{out:#?}");
        assert_eq!(out[0]["text"], "1024x768");
        assert_ne!(out[0]["text"], MOVED_NOTE);
    }

    /// ⛔ Only a `function_call_output` is rewritten. An image the USER sent is already
    /// in a legal position, and moving it would reorder the operator's own conversation.
    #[test]
    fn an_image_the_user_sent_is_left_exactly_where_it_is() {
        let original = body(json!([
            {"type": "message", "role": "user", "content": [image("u")]},
        ]));
        let mut subject = original.clone();
        ResponsesToolImagePolicy::Rehome.normalize(&mut subject);
        assert_eq!(subject, original);
    }

    /// Several images in one result travel together, in order, in one user message.
    #[test]
    fn every_image_in_one_result_moves_together_and_keeps_its_order() {
        let mut subject = body(json!([tool_output(json!([
            image("first"),
            {"type": "input_text", "text": "between"},
            image("second"),
        ]))]));
        ResponsesToolImagePolicy::Rehome.normalize(&mut subject);
        let input = subject["input"].as_array().expect("array");

        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["output"][0]["text"], "between");
        assert_eq!(input[1]["content"][0]["image_url"], "first");
        assert_eq!(input[1]["content"][1]["image_url"], "second");
    }

    /// Two tool results each get their OWN following message — not one pooled at the end.
    #[test]
    fn two_results_each_get_their_own_following_message() {
        let mut subject = body(json!([
            tool_output(json!([image("a")])),
            tool_output(json!([image("b")])),
        ]));
        ResponsesToolImagePolicy::Rehome.normalize(&mut subject);
        let input = subject["input"].as_array().expect("array");

        assert_eq!(input.len(), 4, "{input:#?}");
        assert_eq!(input[1]["content"][0]["image_url"], "a");
        assert_eq!(input[3]["content"][0]["image_url"], "b");
    }

    /// ⚠ A body this policy does not recognise must pass through untouched rather than
    /// panic — it is applied to every request on its client, including shapes that
    /// predate it.
    #[test]
    fn an_unrecognised_body_is_left_alone_rather_than_panicking() {
        for shape in [
            json!({"model": "m"}),
            json!({"model": "m", "input": "a bare string"}),
            body(json!([{"type": "function_call_output", "output": "text, not blocks"}])),
            body(json!(["not an object"])),
        ] {
            let mut subject = shape.clone();
            ResponsesToolImagePolicy::Rehome.normalize(&mut subject);
            assert_eq!(subject, shape, "left alone: {shape}");
        }
    }

    /// A tool result with no image is not rewritten and keeps every block it had.
    #[test]
    fn a_result_without_images_is_untouched() {
        let original = body(json!([tool_output(json!([
            {"type": "input_text", "text": "one"},
            {"type": "input_text", "text": "two"},
        ]))]));
        let mut subject = original.clone();
        ResponsesToolImagePolicy::Rehome.normalize(&mut subject);
        assert_eq!(subject, original);
    }
}
