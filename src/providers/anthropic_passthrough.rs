//! Forwarding to local servers that speak the Anthropic Messages API
//! themselves (llama.cpp's `llama-server`, LM Studio), so no translation is
//! needed: the request goes out with the upstream's model name and the
//! answer, streamed or not, comes back unchanged.

use axum::{
    body::Body,
    http::{StatusCode, header::CONTENT_TYPE},
    response::Response,
};
use serde_json::Value;

use crate::{
    anthropic::{json_error, schema::MessagesRequest},
    provider::RequestContext,
};

/// Sends `body` with `request`, which already carries the URL and any
/// credential headers, under the upstream model name `upstream_model`.
/// `server` names the upstream in error messages.
pub(crate) async fn forward(
    request: reqwest::RequestBuilder,
    mut body: MessagesRequest,
    upstream_model: &str,
    ctx: &RequestContext,
    server: &str,
) -> Response {
    body.model = Some(upstream_model.into());
    let mut payload = serde_json::json!(body);
    // Token counters reject a null max_tokens; an unset one must be omitted.
    if body.max_tokens.is_none()
        && let Value::Object(fields) = &mut payload
    {
        fields.remove("max_tokens");
    }
    if let Some(monitor) = &ctx.monitor {
        monitor.model_resolved(&ctx.req_id, upstream_model);
        monitor.upstream_started(&ctx.req_id);
    }
    let upstream = match request
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
                format!("Could not reach {server}: {error}"),
            );
        }
    };
    if upstream.status().is_redirection() {
        return json_error(
            StatusCode::BAD_GATEWAY,
            "api_error",
            format!("{server} returned a redirect"),
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
