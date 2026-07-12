use serde::Serialize;
use std::fs::File;
use std::io::Write;
use rust::rnet::{RNet};
use rust::drn_agent::DRNAgent;
use candle_core::{Device, Result, Tensor};

#[derive(Serialize)]
struct ProbFrame {
    row: isize,
    col: isize,
    has_mid: bool,
    probabilities: Vec<f32>, // Q値の代わりに選択確率を格納
    best_action: String,
}

pub fn export_probabilities_to_json(model: &RNet, path: &str, device: &Device) -> Result<()> {
    let mut data = Vec::new();
    let temperature: f32 = 0.1; // エージェントのtemp設定に合わせる

    for &mid_flag in &[-1.0, 1.0] {
        for r in 0..5 {
            for c in 0..5 {
                let row_norm = (r as f32 - 2.0) / 2.0;
                let col_norm = (c as f32 - 2.0) / 2.0;
                let input = Tensor::from_slice(&[row_norm, col_norm, mid_flag], (1, 3), device)?;
                
                // 1. モデルから生の値（logits / R値）を取得
                let r_values_tensor = model.forward(&input)?;

                // 2. 温度の適用 (温度が1.0でない場合、値を割る)
                let scaled_tensor = if temperature != 1.0 {
                    r_values_tensor.affine((-1.0 / temperature) as f64, 0.0)?
                } else {
                    r_values_tensor.affine(-1.0,0.0)?
                };

                // 3. ソフトマックス関数で確率分布に変換
                let probs_tensor = candle_nn::ops::softmax(&scaled_tensor, 1)?;
                let probabilities = probs_tensor.flatten_all()?.to_vec1::<f32>()?;

                // 4. 最も確率の高い（＝元のR値が最も低い）行動を選択
                let best_idx = probabilities
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();

                let action_name = match best_idx {
                    0 => "Up",
                    1 => "Down",
                    2 => "Left",
                    3 => "Right",
                    _ => "Unknown",
                };

                data.push(ProbFrame {
                    row: r,
                    col: c,
                    has_mid: mid_flag > 0.5,
                    probabilities,
                    best_action: action_name.to_string(),
                });
            }
        }
    }

    let json = serde_json::to_string_pretty(&data).map_err(candle_core::Error::wrap)?;
    let mut file = File::create(path).map_err(candle_core::Error::wrap)?;

    file.write_all(json.as_bytes()).map_err(candle_core::Error::wrap)?;

    println!("Probability table exported to: {}", path);
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let mut agent = DRNAgent::new(1000, 3);

    // 1. ロード前の値をメモ
    let val_before = {
        let vars = agent.varmap.data().lock().unwrap();
        vars.get("policy.ln1.weight").unwrap().as_tensor().to_vec2::<f32>()?[0][0]
    };

    // 2. ロード実行
    agent.load("checkpoints/ep20.safetensors")?;
    agent.set_lambda(1.0);

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

    // 出力先を reg_prob.json に変更
    let output_path = "reg_prob.json";
    export_probabilities_to_json(&agent.regret_net, output_path, &device)?;

    // テスト：(0,0) と (4,4) で結果が変わるか？
    for test_pos in &[[0.0f32, 0.0, 0.0], [3.0f32, 3.0, 0.0]] {
        let t = Tensor::from_slice(&test_pos[..], (1, 3), &device)?;
        let res = agent.regret_net.forward(&t)?;
        println!("Pos {:?}: {:?}", test_pos, res.to_vec2::<f32>()?);
    }

    Ok(())
}