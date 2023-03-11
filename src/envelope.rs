use nih_plug::prelude::{Smoother, SmoothingStyle};

use crate::MAX_BLOCK_SIZE;

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum ADSRState {
    ATTACK,
    DECAY,
    SUSTAIN,
    RELEASE,
    OFF,
}

struct EnvelopeParams {
    attack_ms: f32,
    decay_ms: f32,
    sustain: f32,
    release_ms: f32,
}
pub struct Envelope {
    pub value: Smoother<f32>,
    state: ADSRState,
}

impl Default for EnvelopeParams {
    fn default() -> Self {
        Self {
            attack_ms: 0.1,
            decay_ms: 1000.0,
            sustain: 0.5,
            release_ms: 500.0,
        }
    }
}

impl Default for Envelope {
    fn default() -> Self {
        Self {
            value: Smoother::none(),
            state: ADSRState::OFF,
        }
    }
}

impl Envelope {
    pub fn process(
        &mut self,
        sample_rate: f32,
        amp_decay_ms: f32,
        amp_release_ms: f32,
        amp_sustain_level: f32,
        block_len: usize,
    ) -> [f32; MAX_BLOCK_SIZE] {
        // nih_log!("Steps left {}", self.value.steps_left());
        let mut out_env = [0.0; MAX_BLOCK_SIZE];
        match self.state {
            ADSRState::OFF => {
                self.value = Smoother::none();
            }
            ADSRState::ATTACK => {
                if self.value.steps_left() <= 0 {
                    self.value = Smoother::new(SmoothingStyle::Exponential(amp_decay_ms));
                    self.value.reset(1.0);
                    self.value.set_target(sample_rate, amp_sustain_level);
                    self.state = ADSRState::DECAY;
                }
            }
            ADSRState::DECAY => {
                if self.value.steps_left() <= 0 {
                    self.value = Smoother::new(SmoothingStyle::None);
                    self.value.reset(amp_sustain_level);
                    self.state = ADSRState::SUSTAIN;
                }
            }
            ADSRState::SUSTAIN => {}
            ADSRState::RELEASE => {
                if self.value.steps_left() <= 0 {
                    self.value = Smoother::new(SmoothingStyle::None);
                    self.value.reset(0.0);
                    self.state = ADSRState::OFF;
                }
            }
        }
        self.value.next_block(&mut out_env, block_len);
        out_env
    }

    pub fn value(&mut self) -> Smoother<f32> {
        self.value.clone()
    }

    pub fn set_value(&mut self, value: Smoother<f32>) {
        self.value = value;
    }

    pub fn set_target(&mut self, sample_rate: f32, target: f32) {
        self.value.set_target(sample_rate, target);
    }

    pub fn reset(&mut self, value: f32) {
        self.value.reset(value);
    }

    pub fn state(&mut self) -> ADSRState {
        self.state
    }

    pub fn set_state(&mut self, state: ADSRState) {
        self.state = state;
    }

    // pub fn set_attack_ms(&mut self, attack_ms: f32) {
    //     self.params.attack_ms = attack_ms;
    // }

    // pub fn set_decay_ms(&mut self, decay_ms: f32) {
    //     self.params.decay_ms = decay_ms;
    // }

    // pub fn set_release_ms(&mut self, release_ms: f32) {
    //     self.params.release_ms = release_ms;
    // }
    // pub fn set_sustain(&mut self, sustain: f32) {
    //     self.params.sustain = sustain;
    // }
}
