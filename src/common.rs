use crate::env::Action;

#[derive(Clone,Debug)]
pub struct Experience {
    pub state:Vec<f32>,
    pub action:Action,
    pub reward:f32,
    pub next_state:Vec<f32>,
    pub done:bool,
    pub next_gamma: f32,
}

impl Experience {
    pub fn new(state:Vec<f32>,action:Action,reward:f32,next_state:Vec<f32>,done:bool,next_gamma:f32) -> Self{
        Self { state, action, reward, next_state, done,next_gamma }
    }
}

