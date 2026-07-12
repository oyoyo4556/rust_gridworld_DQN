use candle_core::Result;
use rust::env::Gridworld;
use rust::drn_agent::DRNAgent;
use std::fs;
use std::path::Path;

fn main() -> Result<()> {

    let save_dir ="checkpoints".to_string();
    if !Path::new(&save_dir).exists() {
        fs::create_dir_all(&save_dir).expect("Failed to create save directory.");
        println!("Created directory: {}",save_dir);
    }

    let mut env = Gridworld::new();
    let mut agent = DRNAgent::new(100000,7);

    let episodes = 200000;
    let batch_size = 32;

    let target_interval = 20;
    let mut count = 0f64;

    //agent.load("ep2000_latest.safetensors")?;

    for episode in 1..=episodes {
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut done = false;
        let mut steps = 0;

        while !done{
            let action = agent.get_action(&state)?;
            let (next_state,reward,is_done) = env.step(action);

            agent.add_experience(state.clone(),action,reward,next_state.clone(),is_done);

            state = next_state;
            total_reward += reward;
            done = is_done; 
            steps += 1;

            if steps % 4000 == 0{
                println!(
                    "Episode:{},steps:{},epsilon:{:.2}",episode,steps,agent.epsilon
                );
            }
        }

        if agent.buffer.len() >= agent.buffer.capacity / 10 {
               let _loss = agent.update(batch_size)?;
        }

        if episode % target_interval == 0 {
            agent.update_target_network(0.1)?;
            agent.update_target_regret_network(0.1)?;
        }
        

        if episode % 100 == 0 {
            println!(
                "Episode {}:Total Reward = {:.2},Epsilon = {:.4}",episode,total_reward,agent.epsilon
            );
        }

        if episode % 1000 ==0 && total_reward >= 0.9 {
            let new_lambda = count * 0.1;
            agent.set_lambda(new_lambda);
            count += 1.0;
        }

        if episode % 20000 == 0 {
            let path = format!("{}/ep{}.safetensors",save_dir,episode/1000);
                agent.save(&path)?;
                println!("Model saved on episode {}",episode);
            agent.save(&format!("ep{}.safetensors",episode))?;
        }
    }

    Ok(()) 
}