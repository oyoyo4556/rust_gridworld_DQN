use std::collections::VecDeque;
use rand::prelude::IndexedRandom;
use crate::common::Experience;

#[derive(Clone,Debug)]

pub struct ReplayBuffer{
    pub buffer: VecDeque<Experience>,
    pub capacity: usize,
}

impl ReplayBuffer{
    pub fn new(capacity: usize) -> Self{
        Self{
            buffer:VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn add(&mut self,exp:Experience) {
            if self.buffer.len() >= self.capacity{
                self.buffer.pop_front();
            }
            self.buffer.push_back(Experience{
                state: exp.state,
                action: exp.action,
                reward: exp.reward,
                next_state: exp.next_state,
                done: exp.done,
                next_gamma: exp.next_gamma,
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