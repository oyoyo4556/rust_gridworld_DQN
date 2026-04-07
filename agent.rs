use crate::env::Action;
use candle_core::{Device,Result,Tensor};
use candle_nn::{AdamW,Optimizer,ParamsAdamW,VarBuilder,VarMap,Module};
use crate::dqn::QNetwork;
use crate::buffer::ReplayBuffer;


pub struct DQNAgent {
    device:Device,
    pub varmap:VarMap,
    pub policy_net:QNetwork,
    pub target_net:QNetwork,
    optimizer:AdamW,
    pub buffer:ReplayBuffer,
    gamma:f32,
    pub epsilon:f64,
    epsilon_min:f64,
    epsilon_decay:f64,
}

impl DQNAgent {
    pub fn new(capacity:usize) -> Result<Self>{
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap,
        candle_core::DType::F32,&device);
        let policy_net = QNetwork::new(vs.pp("policy"))?;
        let target_net = QNetwork::new(vs.pp("target"))?;

        let my_params = ParamsAdamW{
            lr:1e-3,
            ..ParamsAdamW::default()
        };

        let optimizer = AdamW::new(varmap.all_vars(),my_params)?;

        Ok(Self { 
            device,
            varmap,
            policy_net,
            target_net,
            optimizer,
            buffer: ReplayBuffer::new(capacity),
            gamma: 0.99,
            epsilon: 1.0,
            epsilon_min: 0.01,
            epsilon_decay: 0.9999, 
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
        if self.buffer.len() < batch_size{
            return Ok(0.0);
        }

        let batch = self.buffer.sample(batch_size);
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

        let q_values = self.policy_net.forward(&states_t)?;
        let current_q = q_values.gather(&actions_t.unsqueeze(1)?,1)?.squeeze(1)?;

        let next_q_values = self.target_net.forward(&next_states_t)?;
        let max_next_q = next_q_values.max(1)?;

        let ones = dones_t.ones_like()?;
        let not_done = ones.sub (&dones_t)?;
        let gamma_t = Tensor::new(self.gamma,&self.device)?;
        let target_q = max_next_q.broadcast_mul(&gamma_t)?.broadcast_mul(&not_done)?.broadcast_add(&rewards_t)?;
        let loss = current_q.sub(&target_q)?.sqr()?.mean_all()?;

        self.optimizer.backward_step(&loss)?;

        if  self.epsilon > self.epsilon_min{
            self.epsilon *= self.epsilon_decay;
        }

        Ok(loss.to_scalar::<f32>()?)
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