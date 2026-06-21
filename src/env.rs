#[derive(Debug,Clone,Copy,PartialEq)]

pub enum Action{
    Up,Down,Left,Right
}

pub struct Gridworld{
    pub size: isize,
    pub agent_pos:(isize,isize),
    pub goal_pos:(isize,isize),
    pub mid_reward_pos:(isize,isize),
    pub has_mid_reward:bool,
}

impl Gridworld{
    pub fn new() -> Self{
        Self{
            size:5,
            agent_pos:(4,0),
            goal_pos:(1,4),
            mid_reward_pos:(2,3),
            has_mid_reward:false,
        }
    }

    pub fn reset(&mut self) -> Vec<f32>{
        self.agent_pos = (4,0);
        self.has_mid_reward = false;
        self.get_state()
    }

    fn is_wall(&self, pos:(isize,isize)) -> bool{
        matches!(pos,(2,1)|(3,1)|(1,3)|(2,4))
    }

    pub fn step(&mut self,action:Action) -> (Vec<f32>,f32,bool) {
        let (mut r,mut c) = self.agent_pos;

        match action {
            Action::Up => r -= 1,
            Action::Down => r += 1,
            Action::Left => c -= 1,
            Action::Right => c += 1,
        }

        if r < 0 || r >= self.size || c < 0 || c >= self.size {
            return (self.get_state(),-0.01,false);
        }

        if self.is_wall((r,c)) {
            return (self.get_state(),-0.1,false);
        }

        self.agent_pos = (r,c);

        if self.agent_pos == self.goal_pos{
            (self.get_state(),1.0,true)
        }
        else if self.agent_pos == self.mid_reward_pos && ! 
        self.has_mid_reward {
            self.has_mid_reward = true;
            (self.get_state(),0.5,false)
        }
        else{
            (self.get_state(),-0.01,false)
        }
    }

    pub fn get_state(&self) -> Vec<f32>{
        let mid_flag = if self.has_mid_reward {1.0} else {-1.0};
        vec![
            (self.agent_pos.0 as f32 - 2.0)/2.0,
            (self.agent_pos.1 as f32 - 2.0)/2.0,
            mid_flag,
        ]
    }
}

impl Action {
    pub fn from_u32(n:u32) -> Self{
        match n {
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            _ => Action::Right,
        }
    }

    pub fn random() -> Self{
        let mut rng = rand::rng();
        match rand::Rng::random_range(&mut rng,0..4){
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            _ => Action::Right,
        }
    }

    pub fn to_u32(&self) -> u32{
        match self{
            Action::Up => 0,
            Action::Down => 1,
            Action::Left => 2,
            Action::Right => 3,
        }
    }
}