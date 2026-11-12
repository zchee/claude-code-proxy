//! Translation between the Anthropic Messages API and the Vertex AI
//! `generateContent` dialect used by Gemini models.

pub mod request;
pub mod stream;
pub mod thought_signature;
