//! Interactive chat over the mmap'd GGUF engine (CPU-only quantized inference).
//!
//! The prompt for each turn is rendered from the model's Jinja chat template
//! (read from the sibling `tokenizer_config.json`), so chat-format-trained
//! models keep their native prompt shape. Responses are sampled greedily or
//! with temperature/top-k/top-p and streamed token-by-token.

use std::io::{self, Write};

use anyhow::{Context, Result};
use minijinja::{Environment, Value};

use crate::data::TokenizerStore;
use crate::generate::GenerateConfig;
use crate::model::gguf::{GgufEngine, GgufKvCache};

/// One entry in the chat transcript.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Template role: `system`, `user`, or `assistant`.
    pub role: String,
    /// The message body.
    pub content: String,
}

/// Render `messages` with a Hugging Face-style Jinja chat template (the value
/// of `tokenizer_config.json`'s `chat_template`). `add_generation_prompt` is
/// set so the output ends where the assistant should start speaking.
pub fn render_chat_template(template: &str, messages: &[ChatMessage]) -> Result<String> {
    let mut env = Environment::new();
    env.add_template("chat", template)
        .map_err(|e| anyhow::anyhow!("failed to compile chat template: {e}"))?;
    let tpl = env
        .get_template("chat")
        .context("failed to resolve compiled chat template")?;
    let msg_values: Vec<Value> = messages
        .iter()
        .map(|m| Value::from_serialize(serde_json::json!({"role": m.role, "content": m.content})))
        .collect();
    let ctx = Value::from_serialize(serde_json::json!({
        "messages": msg_values,
        "add_generation_prompt": true,
        // Some templates interpolate token placeholders; offer them as empty
        // strings so those render instead of hard-failing on `UndefinedError`.
        "bos_token": "",
        "eos_token": "",
        "tools": [],
        "tool_calls": [],
    }));
    tpl.render(ctx).context("failed to render chat template")
}

/// A chat session bound to a GGUF engine, keeping an in-memory transcript so
/// each turn re-renders the full prompt (the model has no cross-turn state).
pub struct GgufChat {
    engine: GgufEngine,
    tokenizer: TokenizerStore,
    config: GenerateConfig,
    messages: Vec<ChatMessage>,
}

impl GgufChat {
    /// Build a session over an already-loaded engine and tokenizer.
    pub fn new(engine: GgufEngine, tokenizer: TokenizerStore, config: GenerateConfig) -> Self {
        Self {
            engine,
            tokenizer,
            config,
            messages: Vec::new(),
        }
    }

    /// The transcript accumulated so far (including the new user turn after
    /// [`respond`](Self::respond)).
    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// The Jinja template this session renders with, if the tokenizer carried one.
    pub fn chat_template(&self) -> Option<&str> {
        self.tokenizer.chat_template.as_deref()
    }

    /// Append one user turn and produce the assistant reply.
    ///
    /// If the tokenizer has a Jinja chat template, the prompt is rendered from
    /// it; otherwise a plain `user: ...\nassistant:` fallback is used.
    pub fn respond(&mut self, user_input: &str) -> Result<String> {
        self.messages.push(ChatMessage {
            role: "user".to_string(),
            content: user_input.to_string(),
        });

        let prompt = match &self.tokenizer.chat_template {
            Some(tpl) => render_chat_template(tpl, &self.messages)?,
            None => {
                let mut s = String::new();
                for m in &self.messages {
                    s.push_str(&format!("{}: {}\n", m.role, m.content));
                }
                s.push_str("assistant:");
                s
            }
        };

        let max_seq = self.engine.config().max_seq_len;
        let mut ids = self
            .tokenizer
            .encode_raw(&prompt)
            .context("failed to tokenize chat prompt")?;
        if ids.is_empty() {
            anyhow::bail!("prompt produced zero tokens");
        }

        let mut cache = GgufKvCache::new(self.engine.config());
        let mut logits = self.engine.forward(&ids, &mut cache, 0)?;

        let mut out_ids: Vec<u32> = Vec::new();
        loop {
            let next = crate::generate::sample_next_token_cpu(&logits, &self.config);
            if next == self.tokenizer.eos_id {
                break;
            }
            if out_ids.len() >= self.config.max_tokens || ids.len() >= max_seq {
                break;
            }
            out_ids.push(next);
            ids.push(next);
            let pos = ids.len() - 1;
            logits = self
                .engine
                .forward(&[next], &mut cache, pos)
                .context("engine step failed")?;
        }

        let text = self
            .tokenizer
            .decode(&out_ids, true)
            .context("failed to decode assistant reply")?;
        self.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: text.clone(),
        });
        Ok(text)
    }
}

/// Print `text` with the characters emitted so far stripped, so a growing
/// decode only prints the new suffix.
struct StreamPrinter {
    printed: usize,
}

impl StreamPrinter {
    fn emit(&mut self, out: &mut dyn Write, text: &str, final_flush: bool) -> io::Result<()> {
        let bytes = text.len();
        if bytes > self.printed {
            let chunk = &text.as_bytes()[self.printed..bytes];
            out.write_all(chunk)?;
            out.flush()?;
            self.printed = bytes;
        } else if final_flush {
            out.flush()?;
        }
        Ok(())
    }
}

/// Run an interactive REPL: read prompts from stdin, stream responses to
/// `stdout`, and stop on EOF or the `exit`/`quit` commands.
pub fn repl(chat: &mut GgufChat) -> Result<()> {
    use std::io::BufRead;
    let template = chat.chat_template();
    let mut stdout = io::stdout().lock();
    writeln!(
        stdout,
        "chat started (engine `{}`, {} params, temp {})",
        chat.engine.name,
        fmt_params(chat.engine.config().n_layers * chat.engine.config().d_model * 2),
        chat.config.temperature,
    )?;
    if template.is_none() {
        writeln!(
            stdout,
            "no chat template found; falling back to plain `user:/assistant:` lines"
        )?;
    }
    writeln!(stdout, "type `exit` or Ctrl-D to quit")?;

    let mut stdin = io::stdin().lock();
    let mut line = String::new();
    loop {
        line.clear();
        stdout.write_all(b"\nuser> ")?;
        stdout.flush()?;
        let n = stdin.read_line(&mut line).context("failed to read stdin")?;
        if n == 0 {
            writeln!(stdout)?;
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if input == "exit" || input == "quit" {
            break;
        }

        chat.messages.push(ChatMessage {
            role: "user".to_string(),
            content: input.to_string(),
        });
        let prompt = match &chat.tokenizer.chat_template {
            Some(tpl) => render_chat_template(tpl, &chat.messages)?,
            None => {
                let mut s = String::new();
                for m in &chat.messages {
                    s.push_str(&format!("{}: {}\n", m.role, m.content));
                }
                s.push_str("assistant:");
                s
            }
        };
        let max_seq = chat.engine.config().max_seq_len;
        let mut ids = chat.tokenizer.encode_raw(&prompt)?;
        let mut cache = GgufKvCache::new(chat.engine.config());
        let mut logits = chat.engine.forward(&ids, &mut cache, 0)?;

        stdout.write_all(b"assistant> ")?;
        let mut printer = StreamPrinter { printed: 0 };
        let mut assembled = Vec::new();
        loop {
            let next = crate::generate::sample_next_token_cpu(&logits, &chat.config);
            if next == chat.tokenizer.eos_id {
                break;
            }
            if assembled.len() >= chat.config.max_tokens || ids.len() >= max_seq {
                break;
            }
            assembled.push(next);
            ids.push(next);
            let pos = ids.len() - 1;
            logits = chat.engine.forward(&[next], &mut cache, pos)?;
            let partial = chat.tokenizer.decode(&assembled, true)?;
            printer.emit(
                &mut stdout,
                &partial,
                assembled.len() == chat.config.max_tokens,
            )?;
        }
        // Flush any trailing decode (byte-offset decode can lag on multi-byte
        // boundaries right before EOS).
        let final_text = chat.tokenizer.decode(&assembled, true)?;
        printer.emit(&mut stdout, &final_text, true)?;
        writeln!(stdout)?;

        chat.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: final_text,
        });
    }
    Ok(())
}

fn fmt_params(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LlmModelConfig;
    use crate::train::TestBackend;

    fn test_template() -> &'static str {
        "{% for message in messages %}{% if message['role'] == 'user' %}<user>{{ message['content'] }}</user>{% else %}<assistant>{{ message['content'] }}</assistant>{% endif %}{% endfor %}<assistant>"
    }

    #[test]
    fn renders_messages_in_order() {
        let rendered = render_chat_template(
            test_template(),
            &[
                ChatMessage {
                    role: "system".into(),
                    content: "be nice".into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: "hi".into(),
                },
            ],
        )
        .unwrap();
        assert!(rendered.contains("<user>hi</user>"), "{rendered}");
        assert!(rendered.ends_with("<assistant>"), "{rendered}");
    }

    #[test]
    fn fallback_prompt_is_plain_lines() {
        use crate::export::export_gguf;

        let config = LlmModelConfig::tiny();
        let device = burn::backend::flex::FlexDevice;
        let model = crate::LlmModel::<TestBackend>::new(&config, &device);
        let dir = tempfile::tempdir().unwrap();
        let tok_path = dir.path().join("tokenizer.json");
        std::fs::write(
            &tok_path,
            r#"{
                "version": "1.0",
                "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "</s>": 3},
                          "unk_token": "[UNK]"},
                "normalizer": null,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": null,
                "decoder": null,
                "added_tokens": []
            }"#,
        )
        .unwrap();
        let tokenizer = TokenizerStore::from_file(&tok_path).unwrap();

        let gguf_path = dir.path().join("model.gguf");
        export_gguf(&model, &config, &tokenizer, &gguf_path, "chat-test").unwrap();
        let engine = GgufEngine::load(&gguf_path).unwrap();

        let mut chat = GgufChat::new(
            engine,
            tokenizer,
            GenerateConfig {
                max_tokens: 8,
                greedy: true,
                ..GenerateConfig::default()
            },
        );
        let _ = chat.respond("hello").unwrap();
        for m in chat.messages() {
            assert!(matches!(m.role.as_str(), "user" | "assistant"));
        }
    }

    #[test]
    fn stream_printer_only_emits_delta() {
        let mut printer = StreamPrinter { printed: 0 };
        let mut buf = Vec::new();
        printer.emit(&mut buf, "hel", false).unwrap();
        printer.emit(&mut buf, "hello", false).unwrap();
        printer.emit(&mut buf, "hello", true).unwrap();
        assert_eq!(std::str::from_utf8(&buf).unwrap(), "hello");
    }
}
