//! Translation between the Anthropic Messages API and the Gemini
//! `generateContent` request and stream that Cloud Code PA
//! (`v1internal:streamGenerateContent`) serves for the Google AI Ultra plan.

pub mod request;
pub mod stream;
pub mod thought_signature;
