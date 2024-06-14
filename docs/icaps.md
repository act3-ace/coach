# ICAPS: Orchestrating autonomous agents: Reinforcement Learning To Hierarchical Planning with COACH

## Introduction:
In many real life applications, semi-autonomous multi-agent systems need to be controlled and evaluated by human operators. Consider the following scenarios: the routing of autonomous vehicles with potential communication interference, coordinated search with quad-copters whose on-board systems execute close quarters navigation faster than goals can be updated, or skill selection for agents engaged in complex coordinated maneuvers in a team-based activity. In all of these cases on-agent systems (control-based or deep learning-based) execute higher-level directed plans in a semi-autonomous fashion, without high resolution control from a directing agent. Such orchestration-style planning problems lie in a fascinating grey area between reinforcement learning, general planning, and human-on-the-loop system design. 

Unfortunately, the technical challenge of coordinating multiple hierarchical agents across multiple systems with complex communication schedules makes for a high barrier for entry for practical research. To facilitate further research in this area, we will guide tutorial participants in using open source tools to easily convert their existing simulations into semi-autonomous planning problems.

This tutorial will provide an introduction to [COACH](https://github.com/act3-ace/coach) - a suite of Python tools for recasting [Gymnasium](https://gymnasium.farama.org/index.html) and [PettingZoo](https://pettingzoo.farama.org/index.html)-compatible Multi-Agent Reinforcement Learning (MARL) problems as orchestration-style planning problems. Traditional Reinforcement Learning (RL) focuses on training low level agents to interact with an environment in a high frequency feedback loop. Once policies have been trained, human direction becomes an orchestration problem, especially with large numbers of agents.

COACH provides tools for researchers to bridge this gap: given any simulation compatible with Gymnasium or PettingZoo, trained agents can be set up to interface with a director agent who tackles high level scheduling, policy selection, or coordination for generating autonomously executed plans. COACH was created in association with the US Air Force Research Laboratory’s Autonomy Capability Team (ACT3).

In this tutorial, we will go over the transformation of a traditional RL problem into a director/semi-autonomous agent style planning problem on an environment chosen from the PettingZoo repository. 

### Tutorial Links:

* [Google Colab](https://colab.research.google.com/drive/1JQ4-zNoQidWD49O3bNBZ8eQYoWSf6-XE?usp=sharing)
* [IPython File](https://github.com/act3-ace/coach/blob/ICAPSDemo/docs/Coach_Tutorial_1_ipython.ipynb)
* [HTML Version - Download Required](https://github.com/act3-ace/coach/blob/ICAPSDemo/docs/Coach_Tutorial_1_ipython.html)

Video from ICAPS - [https://www.youtube.com/playlist?list=PLj-ZdQ5rfSEpKaxfUJ3CaJV8NBi-cdYwI]


*This tutorial will be presented by Dr. Nate Bade. Dr. Bade is an award-winning educator and former teaching professor and program director of the MS in Applied Mathematics (MSAM) program at Northeastern University. He specialized in project based education and designed the MSAM’s graduate machine learning program. A sample curriculum is available at [on github](https://tipthederiver.github.io/Math-7243-2020/index.html). He is currently a Senior Data Scientist at Mobius Logic and works in coordination with ACT3 on hierarchical methods in multi-agent reinforcement learning and automated planning.*
