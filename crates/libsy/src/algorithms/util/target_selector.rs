// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic target selection from a validated JSON classifier verdict.

use std::collections::BTreeMap;

use jsonptr::PointerBuf;
use serde_json::Value;

use super::llm_judge::JudgePolicy;
use crate::core::classifier::{Classification, Score};
use crate::{LibsyError, Result};
use switchyard_protocol::ModelId;

/// Maps one string field in a validated verdict to a configured routing target.
pub(crate) struct TargetSelectorPolicy {
    selector: PointerBuf,
    targets: BTreeMap<String, ModelId>,
}

impl TargetSelectorPolicy {
    /// Parses a JSON Pointer used to read validated verdicts.
    pub(crate) fn new(
        selector: impl Into<String>,
        targets: BTreeMap<String, ModelId>,
    ) -> Result<Self> {
        let selector =
            PointerBuf::parse(selector.into()).map_err(|error| LibsyError::AlgorithmError {
                message: format!("policy selector is not a valid JSON Pointer: {error}"),
            })?;
        if selector.is_root() {
            return Err(LibsyError::AlgorithmError {
                message: "policy selector must identify a response field".to_string(),
            });
        }
        Ok(Self { selector, targets })
    }
}

impl JudgePolicy for TargetSelectorPolicy {
    type Verdict = Value;

    fn to_classification(&self, verdict: Option<&Self::Verdict>) -> Classification {
        let label = verdict
            .and_then(|verdict| self.selector.resolve(verdict).ok())
            .and_then(Value::as_str);
        let target = label.and_then(|label| self.targets.get(label));

        // ⭐ THE VERDICT IS THE ONLY PLACE THE JUDGE EXPLAINS ITSELF, and until now it
        // was dropped right here. A custom-mode schema usually carries a reason beside
        // the routed label — mini-team's is `why` — and nothing downstream ever saw it:
        // the routing log writes one row per completed RESPONSE (route, model, tier,
        // tokens), so *which* member served is observable and *why that member* is not.
        // With a judge biased toward one target, "it picked local again" and "it never
        // understood the request" look identical without this line.
        //
        // ⚠ The WHOLE verdict, not a named field: this policy knows the selector and
        // nothing else about the schema, so picking out `why` would be guessing at one
        // deployment's shape. `label` is what the pointer resolved to, `routed` says
        // whether it matched a configured target — an unroutable label is exactly the
        // silent fail-open case that is otherwise invisible.
        tracing::info!(
            label = label.unwrap_or("<unresolved>"),
            routed = target.is_some(),
            verdict = %verdict.map_or_else(String::new, ToString::to_string),
            "custom classifier verdict"
        );

        match target {
            Some(target) => Classification::Scores(vec![Score {
                target: target.clone(),
                confidence: 1.0,
            }]),
            None => Classification::Ambiguous(vec![]),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::Result;

    #[test]
    fn a_verdict_selects_its_mapped_target() -> Result<()> {
        let policy = TargetSelectorPolicy::new(
            "/decision/target",
            BTreeMap::from([
                ("opus".to_string(), ModelId::from("model/opus")),
                ("sonnet".to_string(), ModelId::from("model/sonnet")),
            ]),
        )?;
        let classification = policy.to_classification(Some(&json!({
            "decision": {"target": "sonnet"}
        })));

        assert_eq!(
            classification.argmax(false)?.map(|score| score.target),
            Some(ModelId::from("model/sonnet"))
        );
        Ok(())
    }

    #[test]
    fn a_missing_or_unknown_target_abstains() -> Result<()> {
        let policy = TargetSelectorPolicy::new(
            "/target",
            BTreeMap::from([("sonnet".to_string(), ModelId::from("model/sonnet"))]),
        )?;

        assert_eq!(
            policy
                .to_classification(Some(&json!({"target": "unknown"})))
                .argmax(false)?,
            None
        );
        assert_eq!(
            policy
                .to_classification(Some(&json!({"reason": "missing"})))
                .argmax(false)?,
            None
        );
        Ok(())
    }

    #[test]
    fn an_invalid_json_pointer_is_rejected() {
        let result = TargetSelectorPolicy::new("/target~2name", BTreeMap::new());
        assert!(matches!(result, Err(LibsyError::AlgorithmError { message })
                if message.contains("valid JSON Pointer")));
    }
}
