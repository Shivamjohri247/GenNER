//! Learning rate scheduling for training
//!
//! Implements common learning rate schedules for fine-tuning.

use genner_core::error::Result;

/// Base trait for learning rate schedules
pub trait LrSchedule: Send + Sync {
    /// Get the learning rate for a given step
    fn get_lr(&self, step: usize) -> f64;

    /// Get the current learning rate (for compatibility)
    fn lr(&self) -> f64 {
        self.get_lr(0)
    }
}

/// Constant learning rate (no scheduling)
#[derive(Clone, Debug)]
pub struct ConstantLr {
    lr: f64,
}

impl ConstantLr {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl LrSchedule for ConstantLr {
    fn get_lr(&self, _step: usize) -> f64 {
        self.lr
    }
}

/// Linear warmup followed by constant learning rate
#[derive(Clone, Debug)]
pub struct WarmupLr {
    warmup_steps: usize,
    base_lr: f64,
    min_lr: f64,
}

impl WarmupLr {
    pub fn new(warmup_steps: usize, base_lr: f64) -> Self {
        Self {
            warmup_steps,
            base_lr,
            min_lr: base_lr / 1000.0,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl LrSchedule for WarmupLr {
    fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup from min_lr to base_lr
            let progress = step as f64 / self.warmup_steps as f64;
            self.min_lr + (self.base_lr - self.min_lr) * progress
        } else {
            self.base_lr
        }
    }
}

/// Cosine annealing with warmup
#[derive(Clone, Debug)]
pub struct CosineAnnealingLr {
    warmup_steps: usize,
    max_steps: usize,
    base_lr: f64,
    min_lr: f64,
}

impl CosineAnnealingLr {
    pub fn new(warmup_steps: usize, max_steps: usize, base_lr: f64) -> Self {
        Self {
            warmup_steps,
            max_steps,
            base_lr,
            min_lr: base_lr / 1000.0,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    fn cosine_decay(&self, step: usize) -> f64 {
        if step >= self.max_steps {
            return self.min_lr;
        }
        let progress = ((step - self.warmup_steps) as f64) / ((self.max_steps - self.warmup_steps) as f64);
        let cosine = 1.0 + (std::f64::consts::PI * progress).cos();
        self.min_lr + (self.base_lr - self.min_lr) * cosine / 2.0
    }
}

impl LrSchedule for CosineAnnealingLr {
    fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            let progress = step as f64 / self.warmup_steps as f64;
            self.min_lr + (self.base_lr - self.min_lr) * progress
        } else {
            self.cosine_decay(step)
        }
    }
}

/// Linear decay with warmup
#[derive(Clone, Debug)]
pub struct LinearDecayLr {
    warmup_steps: usize,
    max_steps: usize,
    base_lr: f64,
    min_lr: f64,
}

impl LinearDecayLr {
    pub fn new(warmup_steps: usize, max_steps: usize, base_lr: f64) -> Self {
        Self {
            warmup_steps,
            max_steps,
            base_lr,
            min_lr: base_lr / 1000.0,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl LrSchedule for LinearDecayLr {
    fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            let progress = step as f64 / self.warmup_steps as f64;
            self.min_lr + (self.base_lr - self.min_lr) * progress
        } else if step < self.max_steps {
            // Linear decay
            let progress = ((step - self.warmup_steps) as f64) / ((self.max_steps - self.warmup_steps) as f64);
            self.base_lr - (self.base_lr - self.min_lr) * progress
        } else {
            self.min_lr
        }
    }
}

/// Exponential decay with warmup
#[derive(Clone, Debug)]
pub struct ExponentialDecayLr {
    warmup_steps: usize,
    base_lr: f64,
    gamma: f64,
    min_lr: f64,
}

impl ExponentialDecayLr {
    pub fn new(warmup_steps: usize, base_lr: f64, gamma: f64) -> Self {
        Self {
            warmup_steps,
            base_lr,
            gamma,
            min_lr: base_lr / 1000.0,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl LrSchedule for ExponentialDecayLr {
    fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            let progress = step as f64 / self.warmup_steps as f64;
            self.min_lr + (self.base_lr - self.min_lr) * progress
        } else {
            // Exponential decay
            let decay_step = (step - self.warmup_steps) as f64;
            let lr = self.base_lr * self.gamma.powf(decay_step);
            lr.max(self.min_lr)
        }
    }
}

/// Inverse square root schedule (used in Transformers)
#[derive(Clone, Debug)]
pub struct InvSqrtLr {
    warmup_steps: usize,
    base_lr: f64,
    min_lr: f64,
}

impl InvSqrtLr {
    pub fn new(warmup_steps: usize, base_lr: f64) -> Self {
        Self {
            warmup_steps,
            base_lr,
            min_lr: base_lr / 1000.0,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl LrSchedule for InvSqrtLr {
    fn get_lr(&self, step: usize) -> f64 {
        if step == 0 {
            return self.min_lr;
        }
        let decay_factor = (self.warmup_steps as f64).sqrt() / (step as f64).sqrt();
        let lr = self.base_lr * decay_factor;
        lr.max(self.min_lr)
    }
}

/// Polynomial decay with warmup
#[derive(Clone, Debug)]
pub struct PolynomialDecayLr {
    warmup_steps: usize,
    max_steps: usize,
    base_lr: f64,
    min_lr: f64,
    power: f64,
}

impl PolynomialDecayLr {
    pub fn new(warmup_steps: usize, max_steps: usize, base_lr: f64) -> Self {
        Self {
            warmup_steps,
            max_steps,
            base_lr,
            min_lr: base_lr / 1000.0,
            power: 1.0, // Linear by default
        }
    }

    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }
}

impl LrSchedule for PolynomialDecayLr {
    fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            let progress = step as f64 / self.warmup_steps as f64;
            self.min_lr + (self.base_lr - self.min_lr) * progress
        } else if step < self.max_steps {
            // Polynomial decay
            let progress = ((step - self.warmup_steps) as f64) / ((self.max_steps - self.warmup_steps) as f64);
            let decay = (1.0 - progress).powf(self.power);
            self.min_lr + (self.base_lr - self.min_lr) * decay
        } else {
            self.min_lr
        }
    }
}

/// Step learning rate (decays by gamma at specified milestones)
#[derive(Clone, Debug)]
pub struct StepLr {
    base_lr: f64,
    gamma: f64,
    milestones: Vec<usize>,
}

impl StepLr {
    pub fn new(base_lr: f64, gamma: f64, milestones: Vec<usize>) -> Self {
        Self {
            base_lr,
            gamma,
            milestones,
        }
    }
}

impl LrSchedule for StepLr {
    fn get_lr(&self, step: usize) -> f64 {
        let num_decays = self.milestones.iter().filter(|&&m| step >= m).count();
        self.base_lr * self.gamma.powi(num_decays as i32)
    }
}

/// Builder for creating common schedules
pub struct LrScheduleBuilder {
    kind: LrScheduleKind,
}

enum LrScheduleKind {
    Constant { lr: f64 },
    Warmup { warmup_steps: usize, base_lr: f64 },
    Cosine { warmup_steps: usize, max_steps: usize, base_lr: f64 },
    Linear { warmup_steps: usize, max_steps: usize, base_lr: f64 },
    Exponential { warmup_steps: usize, base_lr: f64, gamma: f64 },
    InvSqrt { warmup_steps: usize, base_lr: f64 },
    Polynomial { warmup_steps: usize, max_steps: usize, base_lr: f64, power: f64 },
    Step { base_lr: f64, gamma: f64, milestones: Vec<usize> },
}

impl LrScheduleBuilder {
    /// Create a constant learning rate schedule
    pub fn constant(lr: f64) -> Self {
        Self {
            kind: LrScheduleKind::Constant { lr },
        }
    }

    /// Create a warmup-only schedule
    pub fn warmup(warmup_steps: usize, base_lr: f64) -> Self {
        Self {
            kind: LrScheduleKind::Warmup { warmup_steps, base_lr },
        }
    }

    /// Create a cosine annealing schedule
    pub fn cosine(warmup_steps: usize, max_steps: usize, base_lr: f64) -> Self {
        Self {
            kind: LrScheduleKind::Cosine { warmup_steps, max_steps, base_lr },
        }
    }

    /// Create a linear decay schedule
    pub fn linear(warmup_steps: usize, max_steps: usize, base_lr: f64) -> Self {
        Self {
            kind: LrScheduleKind::Linear { warmup_steps, max_steps, base_lr },
        }
    }

    /// Create an exponential decay schedule
    pub fn exponential(warmup_steps: usize, base_lr: f64, gamma: f64) -> Self {
        Self {
            kind: LrScheduleKind::Exponential { warmup_steps, base_lr, gamma },
        }
    }

    /// Create an inverse sqrt schedule (Transformers-style)
    pub fn inv_sqrt(warmup_steps: usize, base_lr: f64) -> Self {
        Self {
            kind: LrScheduleKind::InvSqrt { warmup_steps, base_lr },
        }
    }

    /// Create a polynomial decay schedule
    pub fn polynomial(warmup_steps: usize, max_steps: usize, base_lr: f64, power: f64) -> Self {
        Self {
            kind: LrScheduleKind::Polynomial { warmup_steps, max_steps, base_lr, power },
        }
    }

    /// Create a step learning rate schedule
    pub fn step(base_lr: f64, gamma: f64, milestones: Vec<usize>) -> Self {
        Self {
            kind: LrScheduleKind::Step { base_lr, gamma, milestones },
        }
    }

    /// Build the schedule
    pub fn build(self) -> Box<dyn LrSchedule> {
        match self.kind {
            LrScheduleKind::Constant { lr } => Box::new(ConstantLr::new(lr)),
            LrScheduleKind::Warmup { warmup_steps, base_lr } => Box::new(WarmupLr::new(warmup_steps, base_lr)),
            LrScheduleKind::Cosine { warmup_steps, max_steps, base_lr } => {
                Box::new(CosineAnnealingLr::new(warmup_steps, max_steps, base_lr))
            }
            LrScheduleKind::Linear { warmup_steps, max_steps, base_lr } => {
                Box::new(LinearDecayLr::new(warmup_steps, max_steps, base_lr))
            }
            LrScheduleKind::Exponential { warmup_steps, base_lr, gamma } => {
                Box::new(ExponentialDecayLr::new(warmup_steps, base_lr, gamma))
            }
            LrScheduleKind::InvSqrt { warmup_steps, base_lr } => {
                Box::new(InvSqrtLr::new(warmup_steps, base_lr))
            }
            LrScheduleKind::Polynomial { warmup_steps, max_steps, base_lr, power } => {
                Box::new(PolynomialDecayLr::new(warmup_steps, max_steps, base_lr).with_power(power))
            }
            LrScheduleKind::Step { base_lr, gamma, milestones } => {
                Box::new(StepLr::new(base_lr, gamma, milestones))
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_lr() {
        let schedule = ConstantLr::new(0.001);
        assert_eq!(schedule.get_lr(0), 0.001);
        assert_eq!(schedule.get_lr(100), 0.001);
        assert_eq!(schedule.get_lr(1000), 0.001);
    }

    #[test]
    fn test_warmup_lr() {
        let schedule = WarmupLr::new(100, 0.001);
        // Before warmup
        assert!(schedule.get_lr(0) > 0.0);
        assert!(schedule.get_lr(0) < 0.001);

        // At end of warmup
        assert_eq!(schedule.get_lr(100), 0.001);

        // After warmup
        assert_eq!(schedule.get_lr(200), 0.001);
    }

    #[test]
    fn test_cosine_annealing_lr() {
        let schedule = CosineAnnealingLr::new(100, 1000, 0.001);
        // During warmup
        assert!(schedule.get_lr(50) > 0.0);
        assert!(schedule.get_lr(50) <= 0.001);

        // At peak (end of warmup)
        assert_eq!(schedule.get_lr(100), 0.001);

        // Mid decay
        let mid_lr = schedule.get_lr(550); // Halfway through decay
        assert!(mid_lr < 0.001);
        assert!(mid_lr > 0.0);

        // At end
        assert!(schedule.get_lr(1000) < 0.001);
        assert!(schedule.get_lr(1000) >= 0.001 / 1000.0);
    }

    #[test]
    fn test_linear_decay_lr() {
        let schedule = LinearDecayLr::new(100, 1000, 0.001);
        // After warmup, before max
        let mid_lr = schedule.get_lr(550);
        assert!(mid_lr < 0.001);
        assert!(mid_lr > 0.0);

        // After max steps
        assert!(schedule.get_lr(2000) < 0.0001);
    }

    #[test]
    fn test_inv_sqrt_lr() {
        let schedule = InvSqrtLr::new(100, 0.001);
        // At warmup step
        assert_eq!(schedule.get_lr(100), 0.001);

        // After warmup, should decay
        assert!(schedule.get_lr(400) < 0.001);
        assert!(schedule.get_lr(400) > 0.0);
    }

    #[test]
    fn test_step_lr() {
        let schedule = StepLr::new(0.001, 0.1, vec![100, 200]);
        assert_eq!(schedule.get_lr(50), 0.001);
        assert!((schedule.get_lr(100) - 0.0001).abs() < 1e-10); // 0.001 * 0.1
        assert!((schedule.get_lr(150) - 0.0001).abs() < 1e-10);
        assert!((schedule.get_lr(200) - 0.00001).abs() < 1e-12); // 0.0001 * 0.1
        assert!((schedule.get_lr(250) - 0.00001).abs() < 1e-12);
    }

    #[test]
    fn test_lr_schedule_builder() {
        let schedule = LrScheduleBuilder::cosine(100, 1000, 0.001).build();
        assert_eq!(schedule.get_lr(100), 0.001);

        let schedule2 = LrScheduleBuilder::constant(0.005).build();
        assert_eq!(schedule2.get_lr(9999), 0.005);
    }
}
