use candle_core::{Result,Tensor};
use candle_nn::{linear,Linear,Module,VarBuilder,Init};

pub struct DuelingQNet{
    ln1:Linear,
    ln2:Linear,
    ln3:Linear,
    value_head:Linear,
    advantage_head:Linear,
}

impl DuelingQNet{
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let ln1 = linear(3,128,vb.pp("ln1"))?;
        let ln2 = linear(128,64,vb.pp("ln2"))?;
        let ln3 = linear(64,64,vb.pp("ln3"))?;
        let value_head = linear(64,1,vb.pp("value_head"))?;
        let advantage_head = linear(64,4,vb.pp("advantage_head"))?;

        Ok(Self {ln1,ln2,ln3,value_head,advantage_head})
    }

    pub fn forward(&self,xs:&Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu()?;
        let xs = self.ln3.forward(&xs)?;
        let x = xs.relu()?;
        let value = self.value_head.forward(&x)?;
        let advantage =self.advantage_head.forward(&x)?;

        let a_mean = advantage.mean_keepdim(1)?;
        let q = value.broadcast_add(&advantage.broadcast_sub(&a_mean)?)?;

        Ok(q)
    }
}

pub fn customlinear(in_dim: usize, out_dim: usize,bias:f64, vb: VarBuilder) -> Result<Linear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;

    let bs = vb.get_with_hints(
        out_dim,
        "bias",
        Init::Const(bias)
    )?;

    Ok(Linear::new(ws, Some(bs)))
}

pub struct RNet {
    ln1:Linear,
    ln2:Linear,
    ln3:Linear,
    regret:Linear,
}

impl RNet{
    pub fn new(vb: VarBuilder) -> Result<Self> {

        let ln1 = linear(3,256,vb.pp("ln1"))?;
        let ln2 = linear(256,128,vb.pp("ln2"))?;
        let ln3 = linear(128,64,vb.pp("ln3"))?;
        let regret = customlinear(64, 4, 3.0, vb.pp("regret"))?;


        Ok(Self {ln1,ln2,ln3,regret})
    }

    pub fn forward(&self,x:&Tensor) -> Result<Tensor> {
        let mut x = self.ln1.forward(x)?;
        x = x.relu()?;
        x = self.ln2.forward(&x)?;
        x = x.relu()?;
        x = self.ln3.forward(&x)?;
        x = x.relu()?;
        x = self.regret.forward(&x)?;
        x = x.gelu()?;

        Ok(x)
    }
}