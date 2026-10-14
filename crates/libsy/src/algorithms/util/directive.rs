// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic routing from an explicit instruction in the user's own words.
//!
//! A directive maps a model group to a regular expression. When the latest user
//! message matches one, that group serves the turn and the judge is never asked:
//! "let workhorse solve this" is an instruction, not something to classify. No
//! match abstains, so the cascade falls through to the judge exactly as before.

use async_trait::async_trait;
use regex::Regex;
use switchyard_protocol::{Category, ContentBlock, Request, Response};

use super::affinity::is_user_turn;
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Classifier, Score};
use crate::{LibsyError, Result};

/// One compiled directive: a pattern and the group it routes to.
struct Directive {
    category: Category,
    pattern: Regex,
}

/// Routes to the first group whose pattern matches the latest user message.
///
/// Patterns are tried in the order given. Only the latest user turn is read, so a
/// directive in an earlier message does not keep steering later turns, and a
/// tool continuation reads the user message that started its turn.
pub(crate) struct DirectiveClassifier {
    directives: Vec<Directive>,
}

impl DirectiveClassifier {
    /// Compiles `(group, pattern)` pairs, rejecting an invalid pattern or the judge group.
    pub(crate) fn new(directives: &[(Category, String)]) -> Result<Self> {
        let directives = directives
            .iter()
            .map(|(category, pattern)| {
                if *category == Category::Judge {
                    return Err(LibsyError::AlgorithmError {
                        message: "a directive cannot route to the judge group".to_string(),
                    });
                }
                let pattern = Regex::new(pattern).map_err(|error| LibsyError::AlgorithmError {
                    message: format!(
                        "directive for {} is not a valid regex: {error}",
                        category.as_str()
                    ),
                })?;
                Ok(Directive {
                    category: category.clone(),
                    pattern,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { directives })
    }

    /// The first directive matching `text`, with the matched span.
    fn find<'a>(&'a self, text: &'a str) -> Option<(&'a Directive, &'a str)> {
        self.directives.iter().find_map(|directive| {
            directive
                .pattern
                .find(text)
                .map(|found| (directive, found.as_str()))
        })
    }
}

/// The text of the latest user turn, or `None` when the request has none.
fn latest_user_text(request: &Request) -> Option<String> {
    let message = request
        .llm_request
        .messages
        .iter()
        .rev()
        .find(|message| is_user_turn(message))?;
    let text = message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    Some(text)
}

#[async_trait]
impl<S: Send> Classifier<S> for DirectiveClassifier {
    async fn score(
        &self,
        _state: &mut S,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<(Classification, Option<Response>)> {
        let abstain = Ok((Classification::Scores(vec![]), None));
        let Some(text) = latest_user_text(request) else {
            return abstain;
        };
        let Some((directive, matched)) = self.find(&text) else {
            return abstain;
        };
        // A group with no runtime model cannot serve; let the judge decide instead.
        let Some(target) = driver.models_for(&directive.category).first().cloned() else {
            tracing::warn!(
                label = directive.category.as_str(),
                "directive matched a group with no runtime model; falling through"
            );
            return abstain;
        };
        tracing::info!(
            label = directive.category.as_str(),
            matched,
            "directive matched; judge skipped"
        );
        driver.set_evidence(serde_json::json!({
            "source": "directive",
            "label": directive.category.as_str(),
            "matched": matched,
        }));
        Ok((
            Classification::Scores(vec![Score {
                target,
                confidence: 1.0,
                category: Some(directive.category.clone()),
            }]),
            None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use switchyard_protocol::{LlmRequest, Message, ModelId, Role, ToolResult};

    use super::*;
    use crate::core::algorithm::RuntimeModels;

    const NAMED: &str = r"(?i)\b(use|let|ask)\s+(the\s+)?(workhorse|luna)\b";

    fn driver() -> Driver {
        Driver::new(
            "test",
            Arc::new(RuntimeModels::new(
                [
                    (
                        "workhorse".parse().unwrap(),
                        vec![ModelId::from("gpt-6-luna")],
                    ),
                    (
                        "hard".parse().unwrap(),
                        vec![ModelId::from("gpt-5.6-terra")],
                    ),
                ]
                .into(),
            )),
        )
        .0
    }

    fn request(messages: Vec<Message>) -> Request {
        Request {
            llm_request: LlmRequest {
                messages,
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    fn classifier(pairs: &[(&str, &str)]) -> DirectiveClassifier {
        let pairs: Vec<(Category, String)> = pairs
            .iter()
            .map(|(group, pattern)| (group.parse().unwrap(), (*pattern).to_string()))
            .collect();
        DirectiveClassifier::new(&pairs).unwrap()
    }

    async fn routed(classifier: &DirectiveClassifier, messages: Vec<Message>) -> Option<ModelId> {
        let (classification, _) = classifier
            .score(&mut (), &mut request(messages), &driver())
            .await
            .unwrap();
        classification
            .argmax(false)
            .unwrap()
            .map(|score| score.target)
    }

    #[tokio::test]
    async fn a_named_role_routes_without_the_judge() {
        let c = classifier(&[("workhorse", NAMED)]);
        let got = routed(
            &c,
            vec![Message::text(Role::User, "let workhorse solve this issue")],
        )
        .await;
        assert_eq!(got, Some(ModelId::from("gpt-6-luna")));
    }

    #[tokio::test]
    async fn a_passing_mention_abstains() {
        let c = classifier(&[("workhorse", NAMED)]);
        for text in [
            "why did workhorse fail?",
            "the workhorse said it is done",
            "fix it",
        ] {
            assert_eq!(
                routed(&c, vec![Message::text(Role::User, text)]).await,
                None,
                "{text}"
            );
        }
    }

    #[tokio::test]
    async fn only_the_latest_user_turn_counts() {
        let c = classifier(&[("workhorse", NAMED)]);
        let got = routed(
            &c,
            vec![
                Message::text(Role::User, "use workhorse for the first part"),
                Message::text(Role::Assistant, "done"),
                Message::text(Role::User, "now review it"),
            ],
        )
        .await;
        assert_eq!(got, None);
    }

    #[tokio::test]
    async fn a_tool_continuation_reads_the_turns_user_message() {
        let c = classifier(&[("workhorse", NAMED)]);
        let tool_result = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "c1".to_string(),
                content: vec![],
                is_error: None,
            })],
        };
        let got = routed(
            &c,
            vec![
                Message::text(Role::User, "Use workhorse and fix implementation"),
                Message::text(Role::Assistant, "running tests"),
                tool_result,
            ],
        )
        .await;
        assert_eq!(got, Some(ModelId::from("gpt-6-luna")));
    }

    #[tokio::test]
    async fn the_first_matching_directive_wins() {
        let c = classifier(&[("hard", r"(?i)\buse\s+hard\b"), ("workhorse", NAMED)]);
        let got = routed(
            &c,
            vec![Message::text(
                Role::User,
                "use hard, or use workhorse if busy",
            )],
        )
        .await;
        assert_eq!(got, Some(ModelId::from("gpt-5.6-terra")));
    }

    #[tokio::test]
    async fn a_group_without_models_falls_through() {
        let c = classifier(&[("planner", r"(?i)\buse\s+planner\b")]);
        let got = routed(&c, vec![Message::text(Role::User, "use planner")]).await;
        assert_eq!(got, None);
    }

    #[test]
    fn the_judge_group_and_bad_patterns_are_rejected() {
        let judge = DirectiveClassifier::new(&[(Category::Judge, "x".to_string())]);
        assert!(matches!(judge, Err(LibsyError::AlgorithmError { message })
                if message.contains("judge")));
        let bad = DirectiveClassifier::new(&[("hard".parse().unwrap(), "(unclosed".to_string())]);
        assert!(matches!(bad, Err(LibsyError::AlgorithmError { message })
                if message.contains("not a valid regex")));
    }
}
