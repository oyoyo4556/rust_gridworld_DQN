use candle_core::{Result,Tensor};
use candle_nn::{linear,Linear,Module,VarBuilder};

pub struct QNetwork {
    ln1:Linear,
    ln2:Linear,
    ln3:Linear,
}

impl QNetwork {
    pub fn new(vs:VarBuilder) -> Result<Self>{
        let ln1 = linear(3,64,vs.pp("ln1"))?;
        let ln2 = linear(64,64,vs.pp("ln2"))?;
        let ln3 = linear(64,4,vs.pp("ln3"))?;

        Ok(Self {ln1,ln2,ln3})
    }
}

impl Module for QNetwork{
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu()?;
        self.ln3.forward(&xs)
    }
}

pub struct DuelingQNet{
    ln1:Linear,
    ln2:Linear,
    value_head:Linear,
    advantage_head:Linear,
}

impl DuelingQNet{
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = linear(3,64,vs.pp("ln1"))?;
        let ln2 = linear(64,64,vs.pp("ln2"))?;
        let value_head = linear(64,1,vs.pp("value_head"))?;
        let advantage_head = linear(64,4,vs.pp("advantage_head"))?;

        Ok(Self {ln1,ln2,value_head,advantage_head})
    }
}

impl Module for DuelingQNet {
    fn forward(&self,xs:&Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        let x = xs.relu()?;
        let value = self.value_head.forward(&x)?;
        let advantage =self.advantage_head.forward(&x)?;

        let a_mean = advantage.mean_keepdim(1)?;
        let q = value.broadcast_add(&advantage.broadcast_sub(&a_mean)?)?;

        Ok(q)
    }
}