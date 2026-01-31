//! Progress reporting utilities

use std::time::Duration;

/// Simple progress reporter
pub struct ProgressReporter {
    total: usize,
    current: usize,
    name: String,
    start_time: std::time::Instant,
    silent: bool,
}

impl ProgressReporter {
    /// Create a new progress reporter
    pub fn new(name: impl Into<String>, total: usize) -> Self {
        Self {
            total,
            current: 0,
            name: name.into(),
            start_time: std::time::Instant::now(),
            silent: false,
        }
    }

    /// Create a silent reporter
    pub fn silent(name: impl Into<String>, total: usize) -> Self {
        Self {
            total,
            current: 0,
            name: name.into(),
            start_time: std::time::Instant::now(),
            silent: true,
        }
    }

    /// Increment progress
    pub fn inc(&mut self, delta: usize) {
        self.current += delta;
        self.print();
    }

    /// Set current progress
    pub fn set(&mut self, current: usize) {
        self.current = current;
        self.print();
    }

    /// Print progress
    fn print(&self) {
        if self.silent {
            return;
        }

        let percent = if self.total > 0 {
            (self.current as f64 / self.total as f64 * 100.0) as u32
        } else {
            0
        };

        let elapsed = self.start_time.elapsed();
        let eta = if self.current > 0 {
            let per_item = elapsed.as_secs_f64() / self.current as f64;
            let remaining = self.total.saturating_sub(self.current);
            Duration::from_secs_f64(per_item * remaining as f64)
        } else {
            Duration::ZERO
        };

        print!(
            "\r{}: {}/{} ({}%) ETA: {:.1}s",
            self.name,
            self.current,
            self.total,
            percent,
            eta.as_secs_f64()
        );

        if self.current >= self.total {
            println!(); // New line when complete
        }
    }

    /// Finish the progress bar
    pub fn finish(mut self) {
        self.current = self.total;
        self.print();
    }
}

impl Drop for ProgressReporter {
    fn drop(&mut self) {
        if self.current < self.total && !self.silent {
            println!(); // Ensure newline on drop
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_reporter() {
        let mut progress = ProgressReporter::silent("test", 100);
        progress.inc(10);
        assert_eq!(progress.current, 10);
        progress.set(50);
        assert_eq!(progress.current, 50);
    }
}
