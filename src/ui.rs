//! A small Ratatui dashboard showing fine-tuning progress.
//!
//! The dashboard runs on a background thread, reads events from a channel, and
//! renders a title bar, a step gauge, the current loss, and a loss-history
//! sparkline. It owns the terminal, so training can happen on the main thread
//! without interference.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Duration;

use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Sparkline};
use ratatui::{Frame, init, restore};

/// A progress event sent from the training loop to the dashboard thread.
enum Event {
    /// A training step completed.
    Step { step: usize, loss: f32 },
    /// Training finished; stop the renderer.
    Done { steps: usize, loss: f32 },
}

/// Handle to the background dashboard.
pub struct Dashboard {
    tx: Sender<Event>,
    handle: Option<JoinHandle<()>>,
}

/// Rendering state shared between the loop and the draw closure.
struct State {
    steps: usize,
    step: usize,
    loss: f32,
    history: Vec<u64>,
    finished: bool,
}

impl Dashboard {
    /// Start the dashboard for a run of `steps` total optimization steps.
    pub fn start(steps: usize) -> Self {
        let (tx, rx) = channel();
        let handle = std::thread::spawn(move || run(rx, steps));
        Self {
            tx,
            handle: Some(handle),
        }
    }

    /// Report the loss of a completed training step.
    pub fn update(&self, step: usize, loss: f32) {
        let _ = self.tx.send(Event::Step { step, loss });
    }

    /// Finish training, flush a final render, and join the renderer thread.
    pub fn finish(mut self, step: usize, loss: f32) {
        let _ = self.tx.send(Event::Done { steps: step, loss });
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Run the renderer until the sender is dropped or a `Done` event arrives.
fn run(rx: Receiver<Event>, steps: usize) {
    let mut state = State {
        steps,
        step: 0,
        loss: 0.0,
        history: Vec::new(),
        finished: false,
    };

    let mut terminal = init();

    while !state.finished {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(Event::Step { step, loss }) => {
                state.step = step;
                state.loss = loss;
                state.history.push((loss * 1000.0) as u64);
                if state.history.len() > 200 {
                    state.history.remove(0);
                }
                let _ = terminal.draw(|frame| draw(frame, &state));
            }
            Ok(Event::Done { steps, loss }) => {
                state.steps = steps;
                state.loss = loss;
                state.finished = true;
                let _ = terminal.draw(|frame| draw(frame, &state));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                let _ = terminal.draw(|frame| draw(frame, &state));
            }
            Err(_) => break,
        }
    }

    restore();
}

/// Render a single frame.
fn draw(frame: &mut Frame<'_>, state: &State) {
    let layout = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Length(4),
        Constraint::Length(3),
        Constraint::Min(0),
    ])
    .split(frame.area());

    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            " llm-burner ",
            Style::new().fg(Color::Green).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            " — Gemma-style transformer fine-tuning",
            Style::new().fg(Color::Gray),
        ),
    ]))
    .block(Block::new().borders(Borders::ALL).title("Training"));

    let percent = if state.steps == 0 {
        0.0
    } else {
        state.step as f64 / state.steps as f64
    };
    let gauge = Gauge::default()
        .block(
            Block::new()
                .borders(Borders::ALL)
                .title(format!("Step {} / {}", state.step, state.steps)),
        )
        .gauge_style(Style::new().fg(Color::Green).bg(Color::DarkGray))
        .percent((percent * 100.0).clamp(0.0, 100.0) as u16)
        .label(format!("{:.1}%", percent * 100.0));

    let loss_para = Paragraph::new(format!("{:.6}", state.loss))
        .block(Block::new().borders(Borders::ALL).title("Loss"))
        .style(Style::new().fg(Color::Cyan));

    let status = if state.finished {
        Paragraph::new("finished")
            .block(Block::new().borders(Borders::ALL).title("Status"))
            .style(Style::new().fg(Color::Green))
    } else {
        Paragraph::new("training…")
            .block(Block::new().borders(Borders::ALL).title("Status"))
            .style(Style::new().fg(Color::Yellow))
    };

    let max = state.history.iter().copied().max().unwrap_or(1).max(1);
    let spark = Sparkline::default()
        .block(Block::new().borders(Borders::ALL).title("Loss history"))
        .data(&state.history)
        .max(max)
        .style(Style::new().fg(Color::Magenta));

    frame.render_widget(title, layout[0]);
    frame.render_widget(gauge, layout[1]);
    frame.render_widget(loss_para, layout[2]);
    frame.render_widget(status, layout[3]);
    frame.render_widget(spark, layout[4]);
}
