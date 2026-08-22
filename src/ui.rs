//! A small Ratatui dashboard showing fine-tuning progress.
//!
//! The dashboard runs on a background thread, reads events from a channel, and
//! renders a title bar, a step gauge, the current loss, and a loss-history
//! sparkline. It owns the terminal while it runs: raw stdout/stderr file
//! descriptors are temporarily redirected into the training log file so stray
//! output from dependencies (cubecl/wgpu compile diagnostics, panics, native
//! libraries) cannot garble the TUI. Rendering itself happens through a saved
//! duplicate of the original stdout descriptor.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Duration;

#[cfg(unix)]
use std::os::unix::io::{FromRawFd, IntoRawFd};

use ratatui::{DefaultTerminal, Frame, Terminal, init, restore};
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::QueueableCommand;
use ratatui::crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Sparkline};

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
    ///
    /// Stdout/stderr are not redirected; prefer
    /// [`Dashboard::start_with_output_redirect`] whenever a log file is
    /// available so library output cannot garble the TUI.
    pub fn start(steps: usize) -> Self {
        Self::start_with_output_redirect(steps, None)
    }

    /// Start the dashboard and redirect raw stdout/stderr writes into
    /// `log_path` (opened in append mode) for as long as the TUI owns the
    /// terminal. The TUI itself renders through a saved duplicate of the
    /// original stdout, so it is unaffected by the redirection.
    pub fn start_with_output_redirect(steps: usize, log_path: Option<&Path>) -> Self {
        let (tx, rx) = channel();
        let log_path = log_path.map(Path::to_path_buf);
        let handle = std::thread::spawn(move || run(rx, steps, log_path));
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
fn run(rx: Receiver<Event>, steps: usize, redirect_log: Option<PathBuf>) {
    let mut state = State {
        steps,
        step: 0,
        loss: 0.0,
        history: Vec::new(),
        finished: false,
    };

    // A private handle to the real terminal, independent of fd 1/2 targets.
    #[cfg(unix)]
    let tty_file = acquire_tty();
    let redirect = redirect_log.as_deref().and_then(|path| {
        match OutputRedirect::open(path) {
            Ok(r) => Some(r),
            Err(err) => {
                eprintln!(
                    "note: could not redirect TUI output into {}: {err}",
                    path.display()
                );
                None
            }
        }
    });

    #[cfg(unix)]
    let mut screen = match tty_file.as_ref() {
        Some(tty) => {
            let term_handle = tty
                .try_clone()
                .expect("duplicated tty handle stays valid");
            match enter_terminal(term_handle) {
                Ok(terminal) => Screen::Tty {
                    terminal,
                    writer: tty.try_clone().expect("duplicated tty handle stays valid"),
                },
                Err(err) => {
                    drop(redirect);
                    panic!("failed to initialize the training dashboard: {err}");
                }
            }
        }
        None => Screen::Default(Box::new(init())),
    };
    #[cfg(not(unix))]
    let mut screen = Screen::Default(Box::new(ratatui::init()));

    while !state.finished {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(Event::Step { step, loss }) => {
                state.step = step;
                state.loss = loss;
                state.history.push((loss * 1000.0) as u64);
                if state.history.len() > 200 {
                    state.history.remove(0);
                }
                screen.draw(&state);
            }
            Ok(Event::Done { steps, loss }) => {
                state.steps = steps;
                state.loss = loss;
                state.finished = true;
                screen.draw(&state);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => screen.draw(&state),
            Err(_) => break,
        }
    }

    // Exit the alternate screen while the TUI still owns the terminal, then
    // hand fd 1/2 back so post-training output reaches the real terminal.
    screen.teardown();
    drop(redirect);
}

/// The active terminal: either rendering through a private tty handle (so fd
/// 1/2 can be redirected elsewhere), or ratatui's default stdout backend.
enum Screen {
    #[cfg(unix)]
    Tty {
        terminal: Terminal<CrosstermBackend<File>>,
        writer: File,
    },
    Default(Box<DefaultTerminal>),
}

impl Screen {
    fn draw(&mut self, state: &State) {
        let result = match self {
            #[cfg(unix)]
            Screen::Tty { terminal, .. } => terminal.draw(|frame| draw(frame, state)),
            Screen::Default(terminal) => terminal.draw(|frame| draw(frame, state)),
        };
        let _ = result;
    }

    /// Leave the alternate screen and raw mode, restoring the terminal state.
    fn teardown(self) {
        use std::io::Write as _;

        match self {
            #[cfg(unix)]
            Screen::Tty { mut writer, .. } => {
                let _ = writer.queue(LeaveAlternateScreen).and_then(|w| w.flush());
                let _ = disable_raw_mode();
            }
            Screen::Default(_) => restore(),
        }
    }
}

#[cfg(unix)]
fn enter_terminal(file: File) -> std::io::Result<Terminal<CrosstermBackend<File>>> {
    use std::io::Write as _;

    enable_raw_mode()?;
    let mut writer = file;
    writer.queue(EnterAlternateScreen)?.flush()?;
    Terminal::new(CrosstermBackend::new(writer))
}

/// Duplicate the original stdout descriptor so rendering keeps working after
/// fd 1 is pointed somewhere else.
#[cfg(unix)]
fn acquire_tty() -> Option<File> {
    let fd = unsafe { libc::dup(1) };
    if fd < 0 {
        return None;
    }
    Some(unsafe { File::from_raw_fd(fd) })
}

/// Redirects the stdout/stderr file descriptors into a log file until dropped.
///
/// This catches every writer — including native code and panics — not just
/// output emitted through the `log` facade.
#[cfg(unix)]
struct OutputRedirect {
    saved_stdout: i32,
    saved_stderr: i32,
}

#[cfg(unix)]
impl OutputRedirect {
    fn open(path: &Path) -> std::io::Result<Self> {
        let log = OpenOptions::new().create(true).append(true).open(path)?;
        let log_fd = log.into_raw_fd();
        let saved_stdout = unsafe { libc::dup(1) };
        let saved_stderr = unsafe { libc::dup(2) };
        if saved_stdout < 0 || saved_stderr < 0 {
            let err = std::io::Error::last_os_error();
            unsafe {
                if saved_stdout >= 0 {
                    libc::close(saved_stdout);
                }
                if saved_stderr >= 0 {
                    libc::close(saved_stderr);
                }
                libc::close(log_fd);
            }
            return Err(err);
        }
        if unsafe { libc::dup2(log_fd, 1) } < 0 || unsafe { libc::dup2(log_fd, 2) } < 0 {
            let err = std::io::Error::last_os_error();
            unsafe {
                libc::close(saved_stdout);
                libc::close(saved_stderr);
                libc::close(log_fd);
            }
            return Err(err);
        }
        if log_fd > 2 {
            unsafe { libc::close(log_fd) };
        }
        Ok(Self {
            saved_stdout,
            saved_stderr,
        })
    }
}

#[cfg(unix)]
impl Drop for OutputRedirect {
    fn drop(&mut self) {
        unsafe {
            libc::dup2(self.saved_stdout, 1);
            libc::dup2(self.saved_stderr, 2);
            libc::close(self.saved_stdout);
            libc::close(self.saved_stderr);
        }
    }
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
