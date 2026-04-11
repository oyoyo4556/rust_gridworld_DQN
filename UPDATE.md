## update項目

1.DoubleDQN
→過大評価を減らす

2.N-step
→遅延報酬を伝播させやすくするため

3.DuelingDQN
→今回はあまり効果ないと思うが、一応実装

4.PER(Prioritized Experience Replay)
→めちゃくちゃ効果がある

## 該当箇所

1.DoubleDQNはagent.rsのtrain_step()でtragetの取り方を変えただけ

2.N-stepは割引計算をしてからReplayBufferにdiscount_rewardとして入れた→agentのn_step_bufferは割引計算のために導入

3.DuelingDQNはQNetのheadを分けた

4.PERはSumTreeを使った専用のbufferを実装。IS_Weightはbufferが計算しだすとややこしくなるので、agentが計算することにした。
