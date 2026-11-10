pub mod auth;
pub mod client;
pub mod count_tokens;
pub mod sanitize;
pub mod translate;
pub mod usage;

use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::anthropic::schema::{CountTokensResponse, MessagesRequest};
use crate::provider::{
    CliHandlers, Generation, GenerationBody, Provider, ProviderError, ProviderErrorKind,
    RequestContext,
};
use crate::request_identity::ConversationIdentity;

use self::auth::cli::MUSE_CLI;
use self::auth::manager::MuseAuthManager;
use self::client::{MuseResponsesClient, anthropic_error_response};
use self::translate::model_allowlist::{MUSE_MODELS, assert_allowed_model};
use self::translate::reader::{MuseReader, ReaderConfig, spawn_reader};
use self::translate::request::{MuseResponsesRequest, translate_request};

pub struct MuseProvider {
    client: OnceLock<Arc<MuseResponsesClient>>,
}

impl MuseProvider {
    /// The server's provider. The HTTP clients are built on the first
    /// request rather than here: building them loads the system root
    /// certificates (about 150 ms each), and every CLI command and server
    /// start constructs every provider. The key is minted on first use too.
    pub fn new() -> Self {
        Self {
            client: OnceLock::new(),
        }
    }

    pub fn with_client(client: MuseResponsesClient) -> Self {
        Self {
            client: OnceLock::from(Arc::new(client)),
        }
    }

    fn client(&self) -> Result<&Arc<MuseResponsesClient>, ProviderError> {
        if let Some(client) = self.client.get() {
            return Ok(client);
        }
        let built = MuseAuthManager::from_config()
            .and_then(|auth| {
                MuseResponsesClient::new(
                    crate::config::muse_base_url()?,
                    crate::config::muse_user_agent()?,
                    Arc::new(auth),
                )
            })
            .map_err(|error| {
                ProviderError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    ProviderErrorKind::Api,
                    format!("Muse transport is unavailable: {error}"),
                )
            })?;
        // Concurrent first requests may each build one; all use the first.
        Ok(self.client.get_or_init(|| Arc::new(built)))
    }

    /// Validates and translates the request, sends it, and starts the reader
    /// task on the response. `/v1/messages` and the OpenAI-compatible routes
    /// share this path, so both get the same translation and error mapping.
    async fn start(
        &self,
        body: &MessagesRequest,
        ctx: &RequestContext,
        conversation_identity: Option<&ConversationIdentity>,
    ) -> Result<(MuseReader, String), ProviderError> {
        let (model, translated) = translate(body, conversation_identity)?;
        let client = self.client()?;
        if let Some(monitor) = &ctx.monitor {
            monitor.model_resolved(&ctx.req_id, &model);
            monitor.upstream_started(&ctx.req_id);
        }
        let upstream = client
            .post(&translated, ctx.traffic.as_deref())
            .await
            .map_err(client::MuseError::into_provider_error)?;
        let reader = spawn_reader(
            upstream.response.bytes_stream(),
            ReaderConfig {
                req_id: ctx.req_id.clone(),
                message_id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
                model: model.clone(),
                estimated_input_tokens: count_tokens::count_tokens(&translated),
                login: upstream.key.login,
                usage_sink: client.auth().clone(),
                secrets: upstream.key.secrets().map(str::to_string).to_vec(),
                traffic: ctx.traffic.clone(),
                monitor: ctx.monitor.clone(),
            },
        );
        Ok((reader, model))
    }
}

impl Default for MuseProvider {
    fn default() -> Self {
        Self::new()
    }
}

fn invalid_request(error: anyhow::Error) -> ProviderError {
    ProviderError::new(
        StatusCode::BAD_REQUEST,
        ProviderErrorKind::InvalidRequest,
        error.to_string(),
    )
}

fn translate(
    body: &MessagesRequest,
    conversation_identity: Option<&ConversationIdentity>,
) -> Result<(String, MuseResponsesRequest), ProviderError> {
    let model = body.model.clone().unwrap_or_default();
    assert_allowed_model(&model).map_err(invalid_request)?;
    let translated =
        translate_request(body, &model, conversation_identity).map_err(invalid_request)?;
    Ok((model, translated))
}

#[async_trait]
impl Provider for MuseProvider {
    fn name(&self) -> &'static str {
        "muse"
    }

    fn supported_models(&self) -> Vec<String> {
        MUSE_MODELS
            .iter()
            .map(|model| (*model).to_string())
            .collect()
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &MUSE_CLI
    }

    async fn handle_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        self.handle_messages_with_conversation_identity(body, ctx, None)
            .await
    }

    async fn handle_messages_with_conversation_identity(
        &self,
        body: MessagesRequest,
        ctx: RequestContext,
        conversation_identity: Option<ConversationIdentity>,
    ) -> Response {
        let (reader, _) = match self
            .start(&body, &ctx, conversation_identity.as_ref())
            .await
        {
            Ok(started) => started,
            Err(error) => return anthropic_error_response(error),
        };
        if body.stream {
            return reader.into_response();
        }
        match reader.into_message().await {
            Ok(message) => (StatusCode::OK, Json(message)).into_response(),
            Err(error) => anthropic_error_response(error.into_provider_error()),
        }
    }

    /// Answered locally: the upstream counting endpoint does not predict the
    /// billed input (it reported 171 tokens for a prompt billed as 14).
    async fn handle_count_tokens(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        let (_, translated) = match translate(&body, None) {
            Ok(translated) => translated,
            Err(error) => return anthropic_error_response(error),
        };
        let tokens = count_tokens::count_tokens(&translated);
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.usage_updated(&ctx.req_id, Some(tokens), None);
        }
        (
            StatusCode::OK,
            Json(CountTokensResponse {
                input_tokens: tokens,
            }),
        )
            .into_response()
    }

    async fn generate_anthropic_stream(
        &self,
        mut body: MessagesRequest,
        ctx: RequestContext,
    ) -> Result<Generation, ProviderError> {
        body.stream = true;
        let (reader, model) = self.start(&body, &ctx, None).await?;
        Ok(Generation {
            body: GenerationBody::LiveSse(reader.into_body()),
            resolved_model: model,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_builds_no_http_client() {
        let provider = MuseProvider::new();
        assert!(provider.client.get().is_none());
        assert_eq!(provider.name(), "muse");
        assert_eq!(provider.supported_models().len(), 5);
    }
}
