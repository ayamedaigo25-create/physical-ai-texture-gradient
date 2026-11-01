# Texture Gradient and Visual Reinforcement Learning

Physical AI 応用編1 最終課題

## 概要
J. J. Gibsonの生態学的視覚論に着想を得て、
環境表面のランダムドット密度がロボットの視覚強化学習に
与える影響を検討する。

## 開発環境
- Robot: CRANE+ V2
- Simulator: Gazebo Fortress
- Middleware: ROS 2 Humble
- Learning: Gymnasium / Stable-Baselines3 PPO
- Policy: MultiInputPolicy
- Observation: RGB 80x80x3 + joint states
- Action: 4 joints + gripper

## 実験条件
| Condition | Dots |
|---|---:|
| none | 0 |
| sparse | 1,000 |
| medium | 2,000 |
| dense | 4,000 |

## 学習
- 128 environment steps per condition
- 7 completed episodes per condition
- Random seed: 42

## 推論
各条件で独立に学習したPPO方策を使用する。
決定論的推論を実施し、最初の20ステップを比較する。

## 結果
各条件で接近行動を確認したが、把持成功には至らなかった。
テクスチャ密度の効果を確かめるには追加試行が必要である。
