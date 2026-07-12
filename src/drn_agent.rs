use crate::env::Action;
use std::cell::RefCell;
use candle_core::{Device,Result,Tensor};
use candle_nn::{AdamW,Optimizer,ParamsAdamW,VarBuilder,VarMap};
use std::collections::VecDeque;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use crate::rnet::{DuelingQNet,RNet};
use crate::buffer::ReplayBuffer;
use crate::common::{Experience};


pub struct DRNAgent {
    device:Device,
    pub varmap:VarMap,
    pub policy_net:DuelingQNet,
    pub target_net:DuelingQNet,
    pub regret_net:RNet,
    pub target_regret_net:RNet,
    q_optimizer:RefCell<AdamW>,
    reg_optimizer:RefCell<AdamW>,
    pub buffer:ReplayBuffer,
    pub epsilon:f64,
    epsilon_min:f64,
    epsilon_decay:f64,
    gamma:f32,
    n_step_buffer:VecDeque<(Vec<f32>,Action,f32,Vec<f32>,bool)>,
    n_step: usize,
    pub lambda:f64,
    temp:f64,
    action_buffer:RefCell<Vec<Action>>,
    weights_buffer:RefCell<Vec<f32>>,
}

impl DRNAgent {
    pub fn new(capacity:usize,n_step:usize) -> Self{
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap,
        candle_core::DType::F32,&device);
        let policy_net = DuelingQNet::new(vb.pp("policy")).unwrap();
        let target_net = DuelingQNet::new(vb.pp("target")).unwrap();
        let regret_net = RNet::new(vb.pp("regret")).unwrap();
        let target_regret_net = RNet::new(vb.pp("target_regret")).unwrap();

        let (q_vars,reg_vars) = {
            let all_vars = varmap.data().lock().map_err(|e|candle_core::Error::Msg(e.to_string())).unwrap();
            let mut q = Vec::new();
            let mut r = Vec::new();

            for (name,var) in all_vars.iter() {
                if name.starts_with("policy.") {
                    q.push(var.clone());
                } else if name.starts_with("regret.") {
                    r.push(var.clone());
                }
            }

            (q,r)
            //ここでdrop
        };

        let my_q_params = ParamsAdamW{
            lr:5e-5,
            ..ParamsAdamW::default()
        };

        let my_reg_params = ParamsAdamW{
            lr:5e-5,
            ..ParamsAdamW::default()
        };

        let q_optimizer = AdamW::new(q_vars,my_q_params).unwrap();
        let reg_optimizer = AdamW::new(reg_vars,my_reg_params).unwrap();


        Self { 
            device,
            varmap,
            policy_net,
            target_net,
            regret_net,
            target_regret_net,
            q_optimizer:RefCell::new(q_optimizer),
            reg_optimizer:RefCell::new(reg_optimizer),
            buffer: ReplayBuffer::new(capacity),
            epsilon:1.0,
            epsilon_min:0.01,
            epsilon_decay:0.9995,
            gamma: 0.99,
            n_step_buffer:VecDeque::with_capacity(n_step),
            n_step, 
            lambda:0.0,
            temp:0.01,
            action_buffer:RefCell::new(Vec::with_capacity(4)),
            weights_buffer:RefCell::new(Vec::with_capacity(4)),
        }
    }

    pub fn get_action(&self, state:&Vec<f32>) -> Result<Action> {
        let mut rng = rand::rng();

        // 1. 値の準備
        let state_tensor = Tensor::from_slice(&state[..],(1,3),&self.device)?;

        if self.lambda == 0.0 {
            if rand::Rng::random_bool(&mut rng,self.epsilon) {
            return Ok(Action::random());
            }

            let q_values = self.policy_net.forward(&state_tensor)?;
            let action_idx = q_values.argmax(1)?.get(0)?.to_scalar::<u32>()?;

            return Ok(Action::from_u32(action_idx));
        }
        
        let q_values = self.policy_net.forward(&state_tensor)?;
        let reg_values = self.regret_net.forward(&state_tensor)?;

        // 2.(1 - λ) * Q - λ * R の計算 (凸結合)
        let lambda_f = self.lambda as f32;
        let scaled_q = q_values.affine((1.0 - lambda_f) as f64, 0.0)?;
        let scaled_r = reg_values.affine(lambda_f as f64, 0.0)?;
        let combined_values = scaled_q.sub(&scaled_r)?;

        // 4. 温度付きsoftmax。
        let temperature = self.temp;
        let scaled_for_softmax = if temperature != 1.0 {
            combined_values.affine(1.0 / temperature, 0.0)?
        } else {
            combined_values
        };

        // 5. ソフトマックス関数で確率分布に変換
        let probs_tensor = candle_nn::ops::softmax(&scaled_for_softmax, 1)?;
        let probs_vec = probs_tensor.flatten_all()?.to_vec1::<f32>()?;

        // 6. 確率分布（重み）に基づいてランダムサンプリング
        // 合法手かつ確率が微小に存在するインデックスと重みを集める
        let mut valid_actions = self.action_buffer.borrow_mut();
        let mut weights = self.weights_buffer.borrow_mut();

        valid_actions.clear();
        weights.clear();

        for (i, &p) in probs_vec.iter().enumerate() {
            if p > 0.0 {
                let action = Action::from_u32(i as u32);
                valid_actions.push(action);
                weights.push(p);
            }
        }

        if valid_actions.is_empty() {
            return Err(candle_core::Error::Msg("No valid legal actions with non-zero probability".to_string()));
        }

        // 重み付きサンプリングの実行
        let dist = WeightedIndex::new(weights.as_slice())
            .map_err(|e| candle_core::Error::Msg(format!("WeightedIndex error: {}", e)))?;
        let chosen_idx = dist.sample(&mut rng);

        Ok(valid_actions[chosen_idx])

        
    }

    pub fn add_experience(&mut self,state:Vec<f32>,action:Action,reward:f32,next_state:Vec<f32>,done:bool) {
        self.n_step_buffer.push_back(
            (state,action,reward,next_state,done)
        );
        if done || self.n_step_buffer.len() >= self.n_step {
            while ! self.n_step_buffer.is_empty(){
                
                let (s_start,a_start,_,_,_) = &self.n_step_buffer[0];
                let mut discount_reward = 0.0;
                for (i,(_,_,r,_,_)) in self.n_step_buffer.iter().enumerate(){
                    discount_reward += r * self.gamma.powi(i as i32);
                }
                let next_gamma = self.gamma.powi(self.n_step_buffer.len() as i32);

                let (_,_,_,last_next_state,last_done) = self.n_step_buffer.back().expect(
                    "Failed to get Last element from n_step_buffer "
                );
                let exp = Experience {
                    state:s_start.clone(),
                    action:*a_start,
                    reward:discount_reward,
                    next_state:last_next_state.clone(),
                    done:*last_done,
                    next_gamma,
                };

                self.buffer.add(exp);

                self.n_step_buffer.pop_front();

                if !done {break;}
            }
        }
    }

    pub fn update(&mut self,batch_size:usize) -> Result<(f32,f32)> {
        if self.buffer.len() < batch_size{
            return Ok((0.0,0.0));
        }

        let batch = self.buffer.sample(batch_size) ;
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
        let next_gammas:Vec<f32> = batch.iter().map(|e| e.next_gamma).collect();
        let next_gammas_t = Tensor::from_vec(next_gammas,batch_size,&self.device)?;

        let actions_t =actions_t.to_dtype(candle_core::DType::U32)?; //gatherするため
        let not_done = (dones_t.ones_like()? - &dones_t)?;
        //=============================================
        // DQN / DuelingQNetの更新
        //=============================================
        let q_values = self.policy_net.forward(&states_t)?;
        let current_q = q_values.gather(&actions_t.unsqueeze(1)?,1)?.squeeze(1)?;

        let next_q_policy = self.policy_net.forward(&next_states_t)?;
        let next_actions = next_q_policy.argmax(1)?;

        let next_q_values = self.target_net.forward(&next_states_t)?;
        let max_next_q = next_q_values.gather(&next_actions.unsqueeze(1)?,1)?.squeeze(1)?;
        let max_next_q = max_next_q.detach();

        let target_q = max_next_q.broadcast_mul(&next_gammas_t)?.broadcast_mul(&not_done)?.broadcast_add(&rewards_t)?;
        let q_loss = candle_nn::loss::huber(&current_q,&target_q,0.4)?;

        let mut q_opt = self.q_optimizer.borrow_mut();

        q_opt.backward_step(&q_loss)?;

        //=============================================
        // DRN / RNetの更新
        //=============================================


        //(A) 現在の予測後悔値R(s,a)の取得
        let r_values = self.regret_net.forward(&states_t)?;
        let current_r = r_values.gather(&actions_t.unsqueeze(1)?,1)?.squeeze(1)?;

        //(B)即時後悔の計算
        let current_q_2 = q_values.detach();
        let max_current_q = current_q_2.max_keepdim(1)?.squeeze(1)?;
        let immediate_regret = max_current_q.sub(&current_q)?;

        //(C)未来の最小後悔値の計算
        let next_r_target = self.target_regret_net.forward(&next_states_t)?;

        let min_next_r = next_r_target.min_keepdim(1)?.squeeze(1)?;
        let min_next_r = min_next_r.detach();

        //(D)Rのtargetの計算
        let target_r = min_next_r.broadcast_mul(&next_gammas_t)?.broadcast_mul(&not_done)?.broadcast_add(&immediate_regret)?;
        let r_loss = candle_nn::loss::huber(&current_r,&target_r,0.1)?;

        let mut reg_opt = self.reg_optimizer.borrow_mut();
        reg_opt.backward_step(&r_loss)?;

        let r_loss_val = r_loss.to_scalar::<f32>()?;

        if  self.epsilon > self.epsilon_min{
            self.epsilon *= self.epsilon_decay;
        }
        

        Ok((q_loss.to_scalar::<f32>()?,r_loss_val))



    }

    pub fn update_target_network(&mut self,tau:f32) -> Result<()> {

        let all_vars = self.varmap.data().lock().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut updates = Vec::new();

        for (name,var) in all_vars.iter(){
            if name.starts_with("policy."){
                let target_name = name.replace("policy.","target.");
                if let Some(target_var) = all_vars.get(&target_name){
                    let p_tensor = var.as_tensor();
                    let t_tensor = target_var.as_tensor();
                    let updated = if tau >= 1.0 {
                        p_tensor.copy()?
                    } else {
                        let t = tau as f64;
                        ((p_tensor * t)? + (t_tensor *(1.0 - t))?)?
                    };

                    updates.push((target_var.clone(),updated));
                    
                }
            }
        }

        drop(all_vars);
        for (var,tensor) in updates {
            var.set(&tensor)?;
        }

        Ok(())
    }

    pub fn update_target_regret_network(&mut self,tau:f32) -> Result<()> {

        let all_vars = self.varmap.data().lock().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut updates = Vec::new();

        for (name,var) in all_vars.iter(){
            if name.starts_with("regret."){
                let target_name = name.replace("regret.","target_regret.");
                if let Some(target_var) = all_vars.get(&target_name){
                    let p_tensor = var.as_tensor();
                    let t_tensor = target_var.as_tensor();
                    let updated = if tau >= 1.0 {
                        p_tensor.copy()?
                    } else {
                        let t = tau as f64;
                        ((p_tensor * t)? + (t_tensor *(1.0 - t))?)?
                    };

                    updates.push((target_var.clone(),updated));
                    
                }
            }
        }

        drop(all_vars);
        for (var,tensor) in updates {
            var.set(&tensor)?;
        }

        Ok(())
    }

    pub fn save(&self,path: &str) -> Result<()>{
        self.varmap.save(path)?;

        Ok(())
    }

    pub fn load(&mut self,path:&str) -> Result<()>{
        self.varmap.load(path)?;
        self.update_target_network(1.0)?;
        self.update_target_regret_network(1.0)?;


        println!("Model loaded from {}",path);

        Ok(())
    }

    pub fn set_qnet_learning_rate(&mut self,lr:f64) {
        self.q_optimizer.borrow_mut().set_learning_rate(lr);
    }

    pub fn set_rnet_learning_rate(&mut self,lr:f64) {
        self.reg_optimizer.borrow_mut().set_learning_rate(lr);
    }

    pub fn copy_weights_to(&self,other:&mut DRNAgent) -> Result<()> {
        //ここでも同様に、全ての更新を一時的にVecに溜めて、ロックを明示的にdropしてから更新する仕様とした。
        let updates = {
            let src_vars = self.varmap.data().lock().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let mut data = Vec::new();
            for (name,var) in src_vars.iter(){
                data.push((name.clone(),var.as_tensor().copy()?));
            }
            data
        };

        {
            let dst_vars = other.varmap.data().lock().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            for (name,tensor) in updates {
                if let Some(dst_var) = dst_vars.get(&name) {
                    dst_var.set(&tensor)?;
                }
            }
        }
        other.update_target_network(1.0)?;
        other.update_target_regret_network(1.0)?;
        Ok(())
    }

    pub fn set_lambda(&mut self,new_lambda:f64) {
        self.lambda = new_lambda.min(1.0);
        println!("Lambda set to: {:.2}", self.lambda);
    }
}