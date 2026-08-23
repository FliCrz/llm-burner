//! Out-of-process GPU capability probe.
//!
//! Compiling a 16-bit-float compute pipeline can hard-crash buggy GPU
//! drivers: Mesa RADV ≤ 25.0 segfaults inside `libvulkan_radeon.so` while
//! creating the first bf16 pipeline on some AMD iGPUs (observed on RENOIR,
//! Mesa 25.0.7). A SIGSEGV raised inside the driver cannot be caught in
//! process, so before committing to a half-precision GPU run we execute a
//! tiny kernel of the requested dtype in a throwaway child process and
//! degrade gracefully (CPU fallback) if the child dies.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::train::Precision;

/// How long the probe child may take to initialize the device, compile, and
/// run its tiny kernel. Cold shader caches make the first run slow; anything
/// beyond this is treated as a hung driver.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(240);

/// How the probe child process terminated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeOutcome {
    /// Kernels were compiled and executed successfully in the requested dtype.
    Success,
    /// The child exited with an error, crashed (driver SIGSEGV), or could not
    /// be spawned. Carries diagnostics for the failure log.
    Failed { detail: String },
    /// The child did not finish within the deadline and was killed.
    TimedOut,
}

impl ProbeOutcome {
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }
}

impl std::fmt::Display for ProbeOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Success => write!(f, "success"),
            Self::TimedOut => write!(f, "timed out after {}s", PROBE_TIMEOUT.as_secs()),
            Self::Failed { detail } => write!(f, "failed: {detail}"),
        }
    }
}

/// Spawn `exe gpu-probe --precision <precision>` and classify termination.
pub fn spawn_probe(exe: &Path, precision: Precision, timeout: Duration) -> ProbeOutcome {
    let precision_arg = precision.to_string();
    spawn_and_classify(
        exe,
        &["gpu-probe", "--precision", precision_arg.as_str()],
        timeout,
    )
}

/// Run `[exe, args..]`, wait up to `timeout`, and classify how it ends.
fn spawn_and_classify(exe: &Path, args: &[&str], timeout: Duration) -> ProbeOutcome {
    let mut child = match Command::new(exe)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            return ProbeOutcome::Failed {
                detail: format!(
                    "could not spawn `{}` {}: {e}",
                    exe.display(),
                    args.join(" ")
                ),
            };
        }
    };

    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut output = String::new();
                if let Some(pipe) = child.stdout.as_mut() {
                    let _ = pipe.read_to_string(&mut output);
                }
                if let Some(pipe) = child.stderr.as_mut() {
                    let _ = pipe.read_to_string(&mut output);
                }
                if status.success() {
                    return ProbeOutcome::Success;
                }
                let reason = describe_exit(status);
                // Keep the last few lines; driver panics put the interesting
                // part at the end of a long log.
                let tail: Vec<&str> = output.lines().collect();
                let tail_start = tail.len().saturating_sub(6);
                let detail = tail[tail_start..].join(" | ");
                return ProbeOutcome::Failed {
                    detail: format!("{reason}; output tail: {detail}"),
                };
            }
            Ok(None) => {
                if started.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return ProbeOutcome::TimedOut;
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(e) => {
                return ProbeOutcome::Failed {
                    detail: format!("waiting on probe failed: {e}"),
                };
            }
        }
    }
}

fn describe_exit(status: std::process::ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return format!("terminated by signal {signal}");
        }
    }
    match status.code() {
        Some(code) => format!("exited with status {code}"),
        None => format!("exited with unknown status ({status})"),
    }
}

/// Execute a minimal kernel in the requested dtype on the GPU backend.
///
/// Runs inside the throwaway probe child (`llm-burner gpu-probe`). Any panic
/// here — including an unwinding dtype-validation error — fails the probe;
/// a driver-level crash kills the process outright. Either way the parent
/// sees a non-success outcome.
#[cfg(feature = "gpu")]
pub fn run_gpu_probe(precision: Precision) -> anyhow::Result<()> {
    use burn::tensor::Tensor;
    use half::{bf16, f16};

    // Elementwise ops are enough to force pipeline compilation — the stage
    // where broken drivers segfault — without triggering matmul autotune
    // benchmarking.
    macro_rules! probe {
        ($elem:ty) => {{
            type B = burn::backend::Wgpu<$elem, i32>;
            let device = Default::default();
            let x = Tensor::<B, 1>::ones([1024], &device) * 2 + 1;
            let sum: f32 = x.sum().into_scalar().into();
            anyhow::ensure!(sum.is_finite(), "probe kernel produced a non-finite result");
        }};
    }

    match precision {
        Precision::F32 => probe!(f32),
        Precision::Bf16 => probe!(bf16),
        Precision::F16 => probe!(f16),
    }
    Ok(())
}

#[cfg(not(feature = "gpu"))]
pub fn run_gpu_probe(_precision: Precision) -> anyhow::Result<()> {
    anyhow::bail!("gpu-probe requires a GPU build (--features gpu)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successful_child_is_success() {
        assert_eq!(
            spawn_and_classify(Path::new("/bin/true"), &[], PROBE_TIMEOUT),
            ProbeOutcome::Success
        );
    }

    #[test]
    fn failing_child_is_failed_with_exit_code() {
        let outcome = spawn_and_classify(Path::new("/bin/false"), &[], PROBE_TIMEOUT);
        match outcome {
            ProbeOutcome::Failed { detail } => {
                assert!(detail.contains("status 1"), "unexpected detail: {detail}")
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn slow_child_times_out_and_is_killed() {
        let outcome = spawn_and_classify(
            Path::new("/bin/sleep"),
            &["30"],
            Duration::from_millis(300),
        );
        assert_eq!(outcome, ProbeOutcome::TimedOut);
    }

    #[test]
    fn missing_executable_is_a_failure_not_a_panic() {
        let outcome = spawn_and_classify(
            Path::new("/nonexistent/llm-burner-probe"),
            &[],
            PROBE_TIMEOUT,
        );
        assert!(matches!(outcome, ProbeOutcome::Failed { .. }));
    }
}
