use candle_core::Result;
use rust::env::Gridworld;
use rust::agent::DQNAgent;

fn main() -> Result<()> {
    let mut env = Gridworld::new();
    let mut agent = DQNAgent::new(10000)?;

    let episodes = 1000;
    let batch_size = 32;
    let target_update_interval = 500;

    agent.load("ep1000_1.safetensors")?;

    for episode in 1..=episodes {
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut done = false;
        let mut steps = 0;

        while !done{
            let action = agent.get_action(&state)?;
            let (next_state,reward,is_done) = env.step(action);

            agent.buffer.add(state.clone(),action,reward,next_state.clone(),is_done);

            let _loss = agent.train_step(batch_size)?;

            state = next_state;
            total_reward += reward;
            done = is_done; 
            steps += 1;

            if steps % 100 == 0{
                println!(
                    "Episode:{},steps:{},epsilon:{:.2}",episode,steps,agent.epsilon
                );
            }
        }

        if episode % target_update_interval == 0 {
            agent.update_target_network()?;
        }

        if episode % 50 == 0 {
            println!(
                "Episode {}:Total Reward = {:.2},Epsilon = {:.4}",episode,total_reward,agent.epsilon
            );
        }

        if episode % 100 == 0 {
            agent.save(&format!("ep{}.safetensors",episode))?;
        }
    }

    Ok(()) 
}