use std::collections::VecDeque;
use crate::env::Action;
use rand::prelude::IndexedRandom;
use crate::common::Experience;

#[derive(Clone,Debug)]

pub struct ReplayBuffer{
    pub buffer: VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer{
    pub fn new(capacity: usize) -> Self{
        Self{
            buffer:VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn add(&mut self,state:Vec<f32>,action:Action,reward:f32,
        next_state:Vec<f32>,done:bool,next_gamma:f32) {
            if self.buffer.len() >= self.capacity{
                self.buffer.pop_front();
            }
            self.buffer.push_back(Experience{
                state,
                action,
                reward,
                next_state,
                done,
                next_gamma,
            });
    }

    pub fn sample(&self,batch_size:usize) -> Vec<Experience>{
        let mut rng = rand::rng();
        let (s1,s2) = self.buffer.as_slices();
        let all_exps:Vec<&Experience> 
        = s1.iter().chain(s2.iter()).collect();
        all_exps.choose_multiple(&mut rng,batch_size).cloned().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}