use std::num::NonZeroU16;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use axum::{
    body::Body,
    http::{StatusCode, header::CONTENT_TYPE},
    response::Response,
};
use reqwest::Client;
use serde_json::Value;

use crate::{
    anthropic::{json_error, schema::MessagesRequest},
    provider::{CliHandlers, Provider, RequestContext},
};

const PUBLIC_MODEL: &str = "local/gguf";
const LLAMA_ALIAS: &str = "gguf";

pub struct LocalProvider {
    upstream: Result<(Client, NonZeroU16)>,
}

impl LocalProvider {
    pub fn configured() -> Option<Self> {
        let port = std::env::var_os("CCP_LLAMA_PORT")?;
        Some(Self::new(&port.to_string_lossy()))
    }

    fn new(port: &str) -> Self {
        let upstream = (|| {
            let port = port
                .parse()
                .context("CCP_LLAMA_PORT must be 1 through 65535")?;
            let client = Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(5))
                .build()?;
            Ok((client, port))
        })();
        Self { upstream }
    }

    async fn forward(
        &self,
        mut body: MessagesRequest,
        ctx: RequestContext,
        path: &str,
    ) -> Response {
        let (client, port) = match &self.upstream {
            Ok(upstream) => upstream,
            Err(error) => {
                return json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    error.to_string(),
                );
            }
        };
        body.model = Some(LLAMA_ALIAS.into());
        let mut payload = serde_json::json!(body);
        // llama-server's token counter expects omitted optional fields, not null.
        if body.max_tokens.is_none()
            && let Value::Object(fields) = &mut payload
        {
            fields.remove("max_tokens");
        }
        if let Some(monitor) = &ctx.monitor {
            monitor.model_resolved(&ctx.req_id, LLAMA_ALIAS);
            monitor.upstream_started(&ctx.req_id);
        }
        let url = format!("http://127.0.0.1:{port}/v1/messages{path}");
        let upstream = match client
            .post(url)
            .header("anthropic-version", "2023-06-01")
            .json(&payload)
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                return json_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    format!("Could not reach local llama-server on port {port}: {error}"),
                );
            }
        };
        if upstream.status().is_redirection() {
            return json_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                "Local llama-server returned a redirect",
            );
        }
        let mut response = Response::new(Body::empty());
        *response.status_mut() = upstream.status();
        if let Some(content_type) = upstream.headers().get(CONTENT_TYPE) {
            response
                .headers_mut()
                .insert(CONTENT_TYPE, content_type.clone());
        }
        *response.body_mut() = Body::from_stream(upstream.bytes_stream());
        response
    }
}

#[async_trait]
impl Provider for LocalProvider {
    fn name(&self) -> &'static str {
        "local"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![PUBLIC_MODEL.into()]
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &LocalAuth
    }

    async fn handle_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        self.forward(body, ctx, "").await
    }

    async fn handle_count_tokens(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        self.forward(body, ctx, "/count_tokens").await
    }
}

struct LocalAuth;

impl CliHandlers for LocalAuth {
    fn login(&self) -> Result<()> {
        anyhow::bail!("Local models do not need a login")
    }
    fn device(&self) -> Result<()> {
        self.login()
    }
    fn status(&self) -> Result<()> {
        Ok(())
    }
    fn logout(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use axum::{Json, Router, http::Request, routing::post};
    use http_body_util::BodyExt;
    use serde_json::json;
    use tower::ServiceExt;

    #[tokio::test]
    async fn local_token_count_preserves_tools_and_omits_unset_max_tokens() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port().to_string();
        let backend = Router::new().route(
            "/v1/messages/count_tokens",
            post(|Json(payload): Json<Value>| async move {
                assert_eq!(payload["model"], LLAMA_ALIAS);
                assert_eq!(payload["system"], "Use a tool");
                assert_eq!(payload["tools"][0]["name"], "get_weather");
                assert_eq!(payload["messages"][0]["content"][0]["type"], "tool_result");
                assert!(payload.get("max_tokens").is_none());
                Json(json!({"input_tokens": 23}))
            }),
        );
        let task = tokio::spawn(async move { axum::serve(listener, backend).await.unwrap() });
        let providers: Vec<Arc<dyn Provider>> = vec![Arc::new(LocalProvider::new(&port))];
        let registry =
            crate::Registry::from_providers(crate::config::AliasProvider::Codex, providers);
        let app = crate::server::app(Arc::new(registry));
        let body = json!({
            "model": PUBLIC_MODEL,
            "system": "Use a tool",
            "tools": [{"name":"get_weather","input_schema":{"type":"object"}}],
            "messages": [{"role":"user","content":[{"type":"tool_result","tool_use_id":"id1","content":"Clear"}]}]
        });
        let request = Request::post("/v1/messages/count_tokens")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes).unwrap(),
            json!({"input_tokens":23})
        );
        task.abort();
    }
}
