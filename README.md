# GPT Reverse Engineering
![Pasted image 20230602193801](https://s2.loli.net/2023/06/02/i6mdSPLDrzKxuo5.png)
## Motivation
- 给一个prompt，GPT语言模型的推理过程，就是对**每个token**，**并行的**，**一层一层的**，**按顺序**算activation，last token的last layer activation接上LM_Head就是output logits了
- 这个过程中，几乎每个activation都是有意义的，因为它们都通过attention机制隐式的为last token提供信息，但不同activation的作用是不同的。
- 如何建模某个特定的activation以及某部分特定的参数的作用呢？

## Token Flow Visualizer
- 模型是什么时候开始知道下一个词要填什么的？把预测过程打开看看！
- 把所有的activation都过一下LM_Head，得到一组token distribution，观察token distribution在activation前进中的发生的变化
- 只关注预测概率变化最大的token子集
	- e.g. 预测“苹果的颜色”时，可能与苹果、颜色相关的token的概率分布才有较大改变，而”汽车“一词的概率全程可能都变化不大，期望只观察那些变化巨大的token分布
- 进一步：每个模块从input中获取了什么信息？mapping到了什么位置？
	- Attention模块可以获取来自其他token的信息，而MLP模块不能
	- 能够引起target token的neighbor token set的概率分布发生较大变化的模块是需要编辑的模块
![](https://s2.loli.net/2023/05/24/OaH3LkKNQAfRgZi.gif)
