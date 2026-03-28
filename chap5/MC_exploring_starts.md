1. 反向计算return

2. visted_pairs的设计：  
(1) 在单个episode里面，同一个state可能被采样多次，因此只取第一次的return  
(2) 在不同episode里面，同一个state也可能被采样多次，那么求平均之后作为该state的q_value
