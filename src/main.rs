use rust::env::Gridworld;
use rust::agent::DQNAgent;
use std::thread;
use std::time::Duration;
use candle_core::Result;

fn main() -> Result<()> {
    let mut world = Gridworld::new();
    let mut agent = DQNAgent::new(10000,3)?;
    let mut state = world.reset();
    let mut total_reward = 0.0; 

    println!("シミュレーション開始🦀");

    for step in 1..= 100 {
        let action = agent.get_action(&state)?;

        let (next_state,reward,done) 
        = world.step(action);

        total_reward += reward;
        let pos = world.agent_pos;
        println!(
            "Step:{:02} | Action:{:?} | Pos: ({},{}) | Reward:{:5.2}",
            step,action,pos.0,pos.1,reward
        );

        state = next_state;

        if done {
            println!("🦀Goal!totalReward:{:.2}",total_reward);
            return Ok(());
        }
    thread::sleep(Duration::from_millis(100));
    }
    println!("goalできずに終了しました");

    Ok(())
}