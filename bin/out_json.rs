use serde::Serialize;
use std::fs::File;
use std::io::Write;
use rust::dqn::DuelingQNet;
use rust::agent::DQNAgent;
use candle_core::{Device,Result,Tensor};
use candle_core::Module;

#[derive(Serialize)]

struct QFrame{
    row:isize,
    col:isize,
    has_mid:bool,
    q_values:Vec<f32>,
    best_action:String,
}

pub fn export_q_table_to_json(model:&DuelingQNet,path:&str,device:&Device) -> Result<()>{
    let mut data = Vec::new();

    for &mid_flag in &[-1.0,1.0]{
        for r in 0..5{
            for c in 0..5{

                let row_norm = (r as f32 - 2.0)/2.0;
                let col_norm = (c as f32 - 2.0)/2.0;
                let input = Tensor::from_slice(&[row_norm as f32,col_norm as f32,mid_flag],(1,3),device)?;
                let q_values_tensor = model.forward(&input)?;
                let q_values = q_values_tensor.flatten_all()?.to_vec1::<f32>()?;
                let best_idx = q_values.iter().enumerate().max_by(|(_,a):&(usize,&f32),(_,b):&(usize,&f32)|a.partial_cmp(b).unwrap()).map(|(i,_)|i).unwrap();
                let action_name = match best_idx {
                    0 => "Up",1 => "Down",2 => "Left",3 => "Right",_ => "Unknown",
                };

                data.push(QFrame{
                    row:r,
                    col:c,
                    has_mid: mid_flag>0.5,
                    q_values,
                    best_action:action_name.to_string(),
                });
            }
        }
    }

    let json = serde_json::to_string_pretty(&data).map_err(candle_core::Error::wrap)?;
    let mut file = File::create(path).map_err(candle_core::Error::wrap)?;

    file.write_all(json.as_bytes()).map_err(candle_core::Error::wrap)?;

    println!("Q-table exported to:{}",path);
    Ok(())
}

fn main() -> Result<()>{
    let device = Device::Cpu;
    let mut agent = DQNAgent::new(1000,3)?;

    
    // 1. ロード前の値をメモ
    let val_before = {
        let vars = agent.varmap.data().lock().unwrap();
        vars.get("policy.ln1.weight").unwrap().as_tensor().to_vec2::<f32>()?[0][0]
    };

    // 2. ロード実行
    agent.load("ep2000.safetensors")?;

    // 3. ロード後の値を比較
    let val_after = {
        let vars = agent.varmap.data().lock().unwrap();
        vars.get("policy.ln1.weight").unwrap().as_tensor().to_vec2::<f32>()?[0][0]
    };

    println!("Before: {}, After: {}", val_before, val_after);

    if val_before == val_after {
        println!("⚠️ 警告: ロード前後で値が変わっていません！ファイル内の名前が一致していない可能性があります。");
    } else {
        println!("✅ ロード成功: 重みが更新されました。");
    }

    let output_path = "q_values.json";

    export_q_table_to_json(&agent.policy_net,output_path,&device)?;


   // テスト：(0,0) と (4,4) で結果が変わるか？
   for test_pos in &[ [0.0f32, 0.0, 0.0], [3.0f32, 3.0, 0.0] ] {
       let t = Tensor::from_slice(&test_pos[..], (1, 3), &device)?;
       let res = agent.policy_net.forward(&t)?;
       println!("Pos {:?}: {:?}", test_pos, res.to_vec2::<f32>()?);
}


    Ok(())
}

