use crate::common::Experience;
use crate::env::Action;
use candle_core::{Device,Result,Tensor};
use candle_nn::{AdamW,Optimizer,ParamsAdamW,VarBuilder,VarMap,Module};
use std::collections::VecDeque;
use crate::dqn::DuelingQNet;
use crate::per_buffer::PrioritizedReplayBuffer;


pub struct DQNAgent {
    device:Device,
    pub varmap:VarMap,
    pub policy_net:DuelingQNet,
    pub target_net:DuelingQNet,
    optimizer:AdamW,
    pub buffer:PrioritizedReplayBuffer,
    gamma:f32,
    pub epsilon:f64,
    epsilon_min:f64,
    epsilon_decay:f64,
    pub beta:f32,
    beta_increment:f32,
    n_step_buffer:VecDeque<(Vec<f32>,Action,f32)>,
    n_step: usize,
}

impl DQNAgent {
    pub fn new(capacity:usize,n_step:usize) -> Result<Self>{
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap,
        candle_core::DType::F32,&device);
        let policy_net = DuelingQNet::new(vs.pp("policy"))?;
        let target_net = DuelingQNet::new(vs.pp("target"))?;

        let my_params = ParamsAdamW{
            lr:5e-5,
            ..ParamsAdamW::default()
        };

        let optimizer = AdamW::new(varmap.all_vars(),my_params)?;

        Ok(Self { 
            device,
            varmap,
            policy_net,
            target_net,
            optimizer,
            buffer: PrioritizedReplayBuffer::new(capacity,0.35),
            gamma: 0.99,
            epsilon: 1.0,
            epsilon_min: 0.01,
            epsilon_decay: 0.99995,
            beta:0.4,
            beta_increment:0.00001,
            n_step_buffer:VecDeque::with_capacity(n_step),
            n_step, 
        })
    }

    pub fn get_action(&mut self,state:&Vec<f32>) -> Result<Action> {
        let mut rng = rand::rng();
        if rand::Rng::random_bool(&mut rng,self.epsilon) {
            return Ok(Action::random());
        }
        let state_tensor = Tensor::from_slice(&state[..],(1,3),&self.device)?;
        let q_values = self.policy_net.forward(&state_tensor)?;
        let action_idx = q_values.argmax(1)?.get(0)?.to_scalar::<u32>()?;

        Ok(Action::from_u32(action_idx))
    }

    pub fn train_step(&mut self,batch_size:usize) -> Result<f32>{
        if self.buffer.size() < batch_size{
            return Ok(0.0);
        }

        let (batch,indices,priorities) = self.buffer.sample(batch_size);
        let states:Vec<f32> = batch.iter().flat_map(|e| e.state.clone()).collect();
        let states_t = Tensor::from_vec(states,(batch_size,3),&self.device)?;
        let next_states:Vec<f32> =batch.iter().flat_map(|e| e.next_state.clone()).collect();
        let next_states_t = Tensor::from_vec(next_states,(batch_size,3),&self.device)?;

        let actions:Vec<u32> = batch.iter().map(|e| e.action.to_u32()).collect();
        let actions_t = Tensor::from_vec(actions,batch_size,&self.device)?;
        let rewards:Vec<f32> = batch.iter().map(|e| e.reward).collect();
        let rewards_t = Tensor::from_vec(rewards,batch_size,&self.device)?;
        let dones:Vec<f32> = batch.iter().map(|e| if e.done {1.0} else {0.0}).collect();
        let dones_t = Tensor::from_vec(dones,batch_size,&self.device)?;

        //IS_Weights

        let total_p = self.buffer.tree.total_priority();
        let n = self.buffer.size() as f32;
        let weights_vec:Vec<f32> = priorities.iter().map(|&p| {
            let prob = p/total_p;
            (1.0/(n*prob)).powf(self.beta)
        }).collect();
        let max_w = weights_vec.iter().cloned().fold(f32::MIN,f32::max);
        let weights_norm: Vec<f32> = weights_vec.iter().map(|w| w/max_w).collect();
        let weights_t = Tensor::from_vec(weights_norm,batch_size,&self.device)?;


        let q_values = self.policy_net.forward(&states_t)?;
        let current_q = q_values.gather(&actions_t.unsqueeze(1)?,1)?.squeeze(1)?;

        let next_q_policy = self.policy_net.forward(&next_states_t)?;
        let next_actions = next_q_policy.argmax(1)?;

        let next_q_values = self.target_net.forward(&next_states_t)?;
        let max_next_q = next_q_values.gather(&next_actions.unsqueeze(1)?,1)?.squeeze(1)?;

        let ones = dones_t.ones_like()?;
        let not_done = ones.sub (&dones_t)?;
        let n_gamma = self.gamma.powi(self.n_step as i32);
        let gamma_t = Tensor::new(n_gamma,&self.device)?;
        let target_q = max_next_q.broadcast_mul(&gamma_t)?.broadcast_mul(&not_done)?.broadcast_add(&rewards_t)?;

        let td_errors = current_q.sub(&target_q)?;
        let squared_errors = td_errors.sqr()?;
        let weighted_errors = squared_errors.broadcast_mul(&weights_t)?; 
        let loss = weighted_errors.mean_all()?;

        self.optimizer.backward_step(&loss)?;

        //bufferのpriority_update

        let errors_abs: Vec<f32> = td_errors.to_vec1()?.iter().map(|e: &f32| e.abs()).collect();
        self.buffer.update_priorities(&indices,&errors_abs);

        if  self.epsilon > self.epsilon_min{
            self.epsilon *= self.epsilon_decay;
        }

        if self.beta < 1.0 {
            self.beta = (self.beta + self.beta_increment).min(1.0);
        }

        Ok(loss.to_scalar::<f32>()?)
    }

    pub fn add_experience(&mut self,state:Vec<f32>,action:Action,reward:f32,next_state:Vec<f32>,done:bool) {
        self.n_step_buffer.push_back((state,action,reward));
        if done || self.n_step_buffer.len() >= self.n_step {
            while ! self.n_step_buffer.is_empty(){
                let (s_start,a_start,_) = self.n_step_buffer[0].clone();
                let mut discount_reward = 0.0;
                for (i,(_,_,r)) in self.n_step_buffer.iter().enumerate(){
                    discount_reward += r * self.gamma.powi(i as i32);
                }

                let exp = Experience {
                    state:s_start,
                    action:a_start,
                    reward:discount_reward,
                    next_state:next_state.clone(),
                    done,
                };

                self.buffer.add(exp);

                self.n_step_buffer.pop_front();

                if !done {break;}
            }
        }
    }

    pub fn update_target_network(&mut self) -> Result<()> {
        let all_vars = self.varmap.data().lock().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut updates = Vec::new();

        for (name,var) in all_vars.iter(){
            if name.starts_with("policy."){
                let target_name = name.replace("policy.","target.");
                if let Some(target_var) = all_vars.get(&target_name){
                    updates.push((target_var.clone(),var.as_tensor().clone()));
                }
            }
        }

        for (target_var,src_tensor) in updates{
            target_var.set(&src_tensor)?;
        }
        Ok(())
    }

    pub fn save(&self,path: &str) -> Result<()>{
        self.varmap.save(path)?;

        Ok(())
    }

    pub fn load(&mut self,path:&str) -> Result<()>{
        self.varmap.load(path)?;
        self.update_target_network()?;


        println!("Model loaded from {}",path);

        Ok(())
    }
}