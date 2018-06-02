# Task---Oriented-Visual-Grounding

This work is extension of the work on [Task-oriented language visual grounding using Deep Reinforcement Learning](https://github.com/devendrachaplot/DeepRL-Grounding)

The original work has a simple attention mechanism and still achieves a very high accuracy in comparison when compared to normal concatention of the image level pixels and the natural language instruction.

In order to see the performance of the attention mechanism of the above work I have spent some time in visualizing it . It can be seen in the following ![video](/Final-video(1).mp4) that the attention mapping is not very good.

Hence our first attempt was made in replacing the attention mechanism . We did so by placing stacked attention ( inspired from stacked attention networks ) instead of the simple gated attention mechanism.

We are now trying to add curiosity along with standard RL learning for the agent.
Since in many real-world scenarios, rewards extrinsic to the agent are extremely sparse, or absent
altogether, curiosity can serve as an intrinsic reward signal to enable the agent to explore its
environment and learn skills that might be useful later in its life.

This work is currently in progres
