use crate::common::Experience;
use rand::Rng;

pub struct SumTree {
    capacity:usize,
    tree:Vec<f32>,
}

impl SumTree {
    pub fn new(capacity:usize) -> Self {
        Self { capacity, tree: vec![0.0; 2 * capacity - 1], }
    }

    pub fn update(&mut self, data_idx:usize,priority:f32) {
        let mut tree_idx = data_idx + self.capacity -1 ;
        let change = priority - self.tree[tree_idx];

        self.tree[tree_idx] = priority;

        while tree_idx > 0 {
            tree_idx = (tree_idx - 1)/2;
            self.tree[tree_idx] += change;
        }
    }

    pub fn get_leaf(&self,mut value:f32) -> usize {
        let mut parent_idx = 0;

        while parent_idx < self.capacity - 1 {
            let left_child_idx = 2 * parent_idx + 1;
            let right_child_idx = left_child_idx + 1;

            if value <= self.tree[left_child_idx] {
                parent_idx = left_child_idx;
            } else {
                value -= self.tree[left_child_idx];
                parent_idx = right_child_idx;
            }
        }
        parent_idx - (self.capacity - 1)
    }

    pub fn total_priority(&self) -> f32 {
        self.tree[0]
    }
}

pub struct PrioritizedReplayBuffer {
    pub capacity:usize,
    pub tree:SumTree,
    buffer:Vec<Option<Experience>>,
    cursor:usize,
    full:bool,
    alpha:f32,
    max_priority:f32,
}

impl PrioritizedReplayBuffer {
    pub fn new(capacity:usize,alpha:f32) -> Self{
        let cap = if capacity.is_power_of_two(){
            capacity
        } else {
            capacity.next_power_of_two()
        };
        //capcityは2の累乗にしないといけない(二分木だから)。capacityに大きすぎるものはいれないでね
        Self {
            capacity:cap,
            tree:SumTree::new(cap),
            buffer:vec![None;cap],
            cursor:0,
            full:false,
            alpha,
            max_priority:1.0,
        }
    }

    pub fn add(&mut self,exp:Experience) {
        let p = self.max_priority.powf(self.alpha);
        self.buffer[self.cursor] = Some(exp);
        self.tree.update(self.cursor,p);

        self.cursor = (self.cursor + 1) % self.capacity;
        if self.cursor == 0 {
            self.full = true;
        }
    }

    pub fn update_priorities(&mut self,indices:&[usize],errors:&[f32]){
        for (&idx,&error) in indices.iter().zip(errors.iter()) {
            let priority = error.abs() + 1e-5;
            if priority > self.max_priority {
                self.max_priority = priority;
            }
            let p_alpha = priority.powf(self.alpha);
            self.tree.update(idx,p_alpha);
        }
    }

    pub fn sample(&self,batch_size:usize) -> (Vec<Experience>,Vec<usize>,Vec<f32>){
        let mut samples = Vec::with_capacity(batch_size);
        let mut indices = Vec::with_capacity(batch_size);
        let mut priorities = Vec::with_capacity(batch_size);

        let segment = self.tree.total_priority()/batch_size as f32;

        for i in 0..batch_size {
            let a = segment * i as f32;
            let b = segment * (i+1) as f32;
            //Noneを引いたときの対策でloop処理
            loop {
                let s = rand::rng().random_range(a..b);
                let data_idx = self.tree.get_leaf(s);

                if let Some(exp) = self.buffer[data_idx].as_ref() {
                    samples.push(exp.clone());
                    indices.push(data_idx);
                    priorities.push(self.tree.tree[data_idx + self.capacity -1]);

                    break;
                }
            }
        }
        (samples,indices,priorities)
    }

    pub fn size(&self) -> usize {
        if self.full {
            self.capacity
        } else {
            self.cursor
        }
    }
}
