## Model Architecture

### Mixture of Experts / Ensemble

Zoph, Barret, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. “ST-MoE: Designing Stable and Transferable Sparse Expert Models.” arXiv, April 29, 2022. [https://doi.org/10.48550/arXiv.2202.08906](https://doi.org/10.48550/arXiv.2202.08906).  
Du, Nan, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, et al. “GLaM: Efficient Scaling of Language Models with Mixture-of-Experts.” arXiv, August 1, 2022. [https://doi.org/10.48550/arXiv.2112.06905](https://doi.org/10.48550/arXiv.2112.06905).

Li, Margaret, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A. Smith, and Luke Zettlemoyer. “Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models.” arXiv, August 5, 2022. [http://arxiv.org/abs/2208.03306](http://arxiv.org/abs/2208.03306).

Mustafa, Basil, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, and Neil Houlsby. “Multimodal Contrastive Learning with LIMoE: The Language-Image Mixture of Experts.” arXiv, June 6, 2022. [https://doi.org/10.48550/arXiv.2206.02770](https://doi.org/10.48550/arXiv.2206.02770).

- [https://ai.googleblog.com/2022/06/limoe-learning-multiple-modalities-with.html](https://ai.googleblog.com/2022/06/limoe-learning-multiple-modalities-with.html)

Shen, Sheng, Le Hou, Yanqi Zhou, Nan Du, Shayne Longpre, Jason Wei, Hyung Won Chung, et al. “Mixture-of-Experts Meets Instruction Tuning:A Winning Combination for Large Language Models.” arXiv, July 5, 2023. [http://arxiv.org/abs/2305.14705](http://arxiv.org/abs/2305.14705).

### Context

Haviv, Adi, Ori Ram, Ofir Press, Peter Izsak, and Omer Levy. “Transformer Language Models without Positional Encodings Still Learn Positional Information.” arXiv, December 5, 2022. [https://doi.org/10.48550/arXiv.2203.16634](https://doi.org/10.48550/arXiv.2203.16634).

Sun, Yutao, Li Dong, Barun Patra, Shuming Ma, Shaohan Huang, Alon Benhaim, Vishrav Chaudhary, Xia Song, and Furu Wei. “A Length-Extrapolatable Transformer.” arXiv, December 20, 2022. [https://doi.org/10.48550/arXiv.2212.10554](https://doi.org/10.48550/arXiv.2212.10554).

Poli, Michael, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, and Christopher Ré. “Hyena Hierarchy: Towards Larger Convolutional Language Models.” arXiv, March 5, 2023. [http://arxiv.org/abs/2302.10866](http://arxiv.org/abs/2302.10866).  
Yu, Lili, Dániel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, and Mike Lewis. “MEGABYTE: Predicting Million-Byte Sequences with Multiscale Transformers.” arXiv, May 19, 2023. [https://doi.org/10.48550/arXiv.2305.07185](https://doi.org/10.48550/arXiv.2305.07185).

Mohtashami, Amirkeivan, and Martin Jaggi. “Landmark Attention: Random-Access Infinite Context Length for Transformers.” arXiv, May 25, 2023. [https://doi.org/10.48550/arXiv.2305.16300](https://doi.org/10.48550/arXiv.2305.16300).

Liu, Hao, and Pieter Abbeel. “Blockwise Parallel Transformer for Long Context Large Models.” arXiv, May 30, 2023. [http://arxiv.org/abs/2305.19370](http://arxiv.org/abs/2305.19370).

Nguyen, Eric, Michael Poli, Marjan Faizi, Armin Thomas, Callum Birch-Sykes, Michael Wornow, Aman Patel, et al. “HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution.” arXiv, June 27, 2023. [https://doi.org/10.48550/arXiv.2306.15794](https://doi.org/10.48550/arXiv.2306.15794).

Chen, Shouyuan, Sherman Wong, Liangjian Chen, and Yuandong Tian. “Extending Context Window of Large Language Models via Positional Interpolation.” arXiv, June 28, 2023. [https://doi.org/10.48550/arXiv.2306.15595](https://doi.org/10.48550/arXiv.2306.15595).  
emozilla. “Dynamically Scaled RoPE Further Increases Performance of Long Context LLaMA with Zero Fine-Tuning.” Reddit Post. R/LocalLLaMA, June 30, 2023. [www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/).  
Ding, Jiayu, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, and Furu Wei. “LongNet: Scaling Transformers to 1,000,000,000 Tokens.” arXiv, July 5, 2023. [https://doi.org/10.48550/arXiv.2307.02486](https://doi.org/10.48550/arXiv.2307.02486).

Tworkowski, Szymon, Konrad Staniszewski, Mikołaj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Miłoś. “Focused Transformer: Contrastive Training for Context Scaling.” arXiv, July 6, 2023. [https://doi.org/10.48550/arXiv.2307.03170](https://doi.org/10.48550/arXiv.2307.03170).

## Sparsity and Efficiency

Child, Rewon, Scott Gray, Alec Radford, and Ilya Sutskever. “Generating Long Sequences with Sparse Transformers.” arXiv, April 23, 2019. [http://arxiv.org/abs/1904.10509](http://arxiv.org/abs/1904.10509).  
DeepSpeed. “DeepSpeed Sparse Attention,” September 8, 2020. [https://www.deepspeed.ai/2020/09/08/sparse-attention.html](https://www.deepspeed.ai/2020/09/08/sparse-attention.html).

Zaheer, Manzil, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, et al. “Big Bird: Transformers for Longer Sequences.” arXiv, January 8, 2021. [https://doi.org/10.48550/arXiv.2007.14062](https://doi.org/10.48550/arXiv.2007.14062).

Jaszczur, Sebastian, Aakanksha Chowdhery, Afroz Mohiuddin, Łukasz Kaiser, Wojciech Gajewski, Henryk Michalewski, and Jonni Kanerva. “Sparse Is Enough in Scaling Transformers.” arXiv, November 24, 2021. [https://doi.org/10.48550/arXiv.2111.12763](https://doi.org/10.48550/arXiv.2111.12763).

Tay, Yi, Mostafa Dehghani, Dara Bahri, and Donald Metzler. “Efficient Transformers: A Survey.” arXiv, March 14, 2022. [https://doi.org/10.48550/arXiv.2009.06732](https://doi.org/10.48550/arXiv.2009.06732).

Fedus, William, Barret Zoph, and Noam Shazeer. “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.” arXiv, June 16, 2022. [http://arxiv.org/abs/2101.03961](http://arxiv.org/abs/2101.03961).

Cho, Sungjun, Seonwoo Min, Jinwoo Kim, Moontae Lee, Honglak Lee, and Seunghoon Hong. “Transformers Meet Stochastic Block Models: Attention with Data-Adaptive Sparsity and Cost.” arXiv, October 27, 2022. [https://doi.org/10.48550/arXiv.2210.15541](https://doi.org/10.48550/arXiv.2210.15541).

Thangarasa, Vithursan, Abhay Gupta, William Marshall, Tianda Li, Kevin Leong, Dennis DeCoste, Sean Lie, and Shreyas Saxena. “SPDF: Sparse Pre-Training and Dense Fine-Tuning for Large Language Models.” arXiv, March 18, 2023. [https://doi.org/10.48550/arXiv.2303.10464](https://doi.org/10.48550/arXiv.2303.10464).  
Zhuang, Bohan, Jing Liu, Zizheng Pan, Haoyu He, Yuetian Weng, and Chunhua Shen. “A Survey on Efficient Training of Transformers.” arXiv, May 3, 2023. [https://doi.org/10.48550/arXiv.2302.01107](https://doi.org/10.48550/arXiv.2302.01107).

Dettmers, Tim, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, and Dan Alistarh. “SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression.” arXiv, June 5, 2023. [https://doi.org/10.48550/arXiv.2306.03078](https://doi.org/10.48550/arXiv.2306.03078).

## MQA

Shazeer, Noam. “Fast Transformer Decoding: One Write-Head Is All You Need.” arXiv, November 5, 2019. [https://doi.org/10.48550/arXiv.1911.02150](https://doi.org/10.48550/arXiv.1911.02150).  
Pope, Reiner, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. “Efficiently Scaling Transformer Inference.” arXiv, November 9, 2022. [http://arxiv.org/abs/2211.05102](http://arxiv.org/abs/2211.05102).

Xu, Yangyang, Xiangtai Li, Haobo Yuan, Yibo Yang, and Lefei Zhang. “Multi-Task Learning with Multi-Query Transformer for Dense Prediction.” arXiv, April 7, 2023. [http://arxiv.org/abs/2205.14354](http://arxiv.org/abs/2205.14354).

Ainslie, Joshua, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. “GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.” arXiv, May 22, 2023. [https://doi.org/10.48550/arXiv.2305.13245](https://doi.org/10.48550/arXiv.2305.13245).

## Activation Functions

Shazeer, Noam. “GLU Variants Improve Transformer.” arXiv, February 12, 2020. [https://doi.org/10.48550/arXiv.2002.05202](https://doi.org/10.48550/arXiv.2002.05202).  
Chowdhery, Aakanksha, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, et al. “PaLM: Scaling Language Modeling with Pathways.” arXiv, October 5, 2022. [http://arxiv.org/abs/2204.02311](http://arxiv.org/abs/2204.02311).

Fang, Haishuo, Ji-Ung Lee, Nafise Sadat Moosavi, and Iryna Gurevych. “Transformers with Learnable Activation Functions.” In Findings of the Association for Computational Linguistics: EACL 2023, 2382–98. Dubrovnik, Croatia: Association for Computational Linguistics, 2023. [https://aclanthology.org/2023.findings-eacl.181](https://aclanthology.org/2023.findings-eacl.181).  
Liu, Hong, Zhiyuan Li, David Hall, Percy Liang, and Tengyu Ma. “Sophia: A Scalable Stochastic Second-Order Optimizer for Language Model Pre-Training.” arXiv, May 23, 2023. [http://arxiv.org/abs/2305.14342](http://arxiv.org/abs/2305.14342).

## Grounding

Guu, Kelvin, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. “REALM: Retrieval-Augmented Language Model Pre-Training.” arXiv, February 10, 2020. [https://doi.org/10.48550/arXiv.2002.08909](https://doi.org/10.48550/arXiv.2002.08909).

Lewis, Patrick, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, et al. “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.” arXiv, April 12, 2021. [https://doi.org/10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401).

Jiang, Zhengbao, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. “Active Retrieval Augmented Generation.” arXiv, May 11, 2023. [https://doi.org/10.48550/arXiv.2305.06983](https://doi.org/10.48550/arXiv.2305.06983).

Ma, Xinbei, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. “Query Rewriting for Retrieval-Augmented Large Language Models.” arXiv, May 23, 2023. [https://doi.org/10.48550/arXiv.2305.14283](https://doi.org/10.48550/arXiv.2305.14283).

Shi, Weijia, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. “REPLUG: Retrieval-Augmented Black-Box Language Models.” arXiv, May 24, 2023. [https://doi.org/10.48550/arXiv.2301.12652](https://doi.org/10.48550/arXiv.2301.12652).

Liu, Jiongnan, Jiajie Jin, Zihan Wang, Jiehan Cheng, Zhicheng Dou, and Ji-Rong Wen. “RETA-LLM: A Retrieval-Augmented Large Language Model Toolkit.” arXiv, June 8, 2023. [https://doi.org/10.48550/arXiv.2306.05212](https://doi.org/10.48550/arXiv.2306.05212).

Wang, Weizhi, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei. “Augmenting Language Models with Long-Term Memory.” arXiv, June 12, 2023. [https://doi.org/10.48550/arXiv.2306.07174](https://doi.org/10.48550/arXiv.2306.07174).

Pan, Shirui, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. “Unifying Large Language Models and Knowledge Graphs: A Roadmap.” arXiv, June 20, 2023. [https://doi.org/10.48550/arXiv.2306.08302](https://doi.org/10.48550/arXiv.2306.08302).

## Training

Liu, Hao, and Pieter Abbeel. “Blockwise Parallel Transformer for Long Context Large Models.” arXiv, May 30, 2023. [http://arxiv.org/abs/2305.19370](http://arxiv.org/abs/2305.19370).

### 

Liu, Zhuang, Zhiqiu Xu, Joseph Jin, Zhiqiang Shen, and Trevor Darrell. “Dropout Reduces Underfitting.” arXiv, May 31, 2023. [https://doi.org/10.48550/arXiv.2303.01500](https://doi.org/10.48550/arXiv.2303.01500).  
### Better Data

autolabel: [https://github.com/refuel-ai/autolabel](https://github.com/refuel-ai/autolabel) - "Autolabel is a Python library to label, clean and enrich text datasets with any Large Language Models (LLM) of your choice." Start 2023-03

Trajanoska, Milena, Riste Stojanov, and Dimitar Trajanov. “Enhancing Knowledge Graph Construction Using Large Language Models.” arXiv, May 8, 2023. [https://doi.org/10.48550/arXiv.2305.04676](https://doi.org/10.48550/arXiv.2305.04676).

Eldan, Ronen, and Yuanzhi Li. “TinyStories: How Small Can Language Models Be and Still Speak Coherent English?” arXiv, May 24, 2023. [https://doi.org/10.48550/arXiv.2305.07759](https://doi.org/10.48550/arXiv.2305.07759).

Xu, Can, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. “WizardLM: Empowering Large Language Models to Follow Complex Instructions.” arXiv, June 10, 2023. [https://doi.org/10.48550/arXiv.2304.12244](https://doi.org/10.48550/arXiv.2304.12244).

Luo, Ziyang, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. “WizardCoder: Empowering Code Large Language Models with Evol-Instruct.” arXiv, June 14, 2023. [https://doi.org/10.48550/arXiv.2306.08568](https://doi.org/10.48550/arXiv.2306.08568).

Gunasekar, Suriya, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, et al. “Textbooks Are All You Need.” arXiv, June 20, 2023. [https://doi.org/10.48550/arXiv.2306.11644](https://doi.org/10.48550/arXiv.2306.11644).  
Lee, Alycia, Brando Miranda, and Sanmi Koyejo. “Beyond Scale: The Diversity Coefficient as a Data Quality Metric Demonstrates LLMs Are Pre-Trained on Formally Diverse Data.” arXiv, June 23, 2023. [https://doi.org/10.48550/arXiv.2306.13840](https://doi.org/10.48550/arXiv.2306.13840).

### Self Supervised Training

Sun, Zhiqing, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang, and Chuang Gan. “Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision.” arXiv, May 4, 2023. [https://doi.org/10.48550/arXiv.2305.03047](https://doi.org/10.48550/arXiv.2305.03047).

Rafailov, Rafael, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. “Direct Preference Optimization: Your Language Model Is Secretly a Reward Model.” arXiv, May 29, 2023. [https://doi.org/10.48550/arXiv.2305.18290](https://doi.org/10.48550/arXiv.2305.18290).

Manikandan, Hariharan, Yiding Jiang, and J. Zico Kolter. “Language Models Are Weak Learners.” arXiv, June 24, 2023. [https://doi.org/10.48550/arXiv.2306.14101](https://doi.org/10.48550/arXiv.2306.14101).

Jain, Neel, Khalid Saifullah, Yuxin Wen, John Kirchenbauer, Manli Shu, Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. “Bring Your Own Data! Self-Supervised Evaluation for Large Language Models.” arXiv, June 29, 2023. [https://doi.org/10.48550/arXiv.2306.13651](https://doi.org/10.48550/arXiv.2306.13651).

Song, Feifan, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, and Houfeng Wang. “Preference Ranking Optimization for Human Alignment.” arXiv, June 30, 2023. [https://doi.org/10.48550/arXiv.2306.17492](https://doi.org/10.48550/arXiv.2306.17492).

### Distillation

Gu, Yuxian, Li Dong, Furu Wei, and Minlie Huang. “Knowledge Distillation of Large Language Models.” arXiv, June 14, 2023. [https://doi.org/10.48550/arXiv.2306.08543](https://doi.org/10.48550/arXiv.2306.08543).

Agarwal, Rishabh, Nino Vieillard, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. “GKD: Generalized Knowledge Distillation for Auto-Regressive Sequence Models.” arXiv, June 23, 2023. [https://doi.org/10.48550/arXiv.2306.13649](https://doi.org/10.48550/arXiv.2306.13649).

## Inference

Long, Jieyi. “Large Language Model Guided Tree-of-Thought.” arXiv, May 14, 2023. [https://doi.org/10.48550/arXiv.2305.08291](https://doi.org/10.48550/arXiv.2305.08291).

Yao, Shunyu, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. “Tree of Thoughts: Deliberate Problem Solving with Large Language Models.” arXiv, May 17, 2023. [https://doi.org/10.48550/arXiv.2305.10601](https://doi.org/10.48550/arXiv.2305.10601).

Wang, Guanzhi, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. “Voyager: An Open-Ended Embodied Agent with Large Language Models.” arXiv, May 25, 2023. [https://doi.org/10.48550/arXiv.2305.16291](https://doi.org/10.48550/arXiv.2305.16291).

Lin, Bill Yuchen, Yicheng Fu, Karina Yang, Prithviraj Ammanabrolu, Faeze Brahman, Shiyu Huang, Chandra Bhagavatula, Yejin Choi, and Xiang Ren. “SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks.” arXiv, May 27, 2023. [https://doi.org/10.48550/arXiv.2305.17390](https://doi.org/10.48550/arXiv.2305.17390).  
Shinn, Noah, Federico Cassano, Beck Labash, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. “Reflexion: Language Agents with Verbal Reinforcement Learning.” arXiv, June 10, 2023. [https://doi.org/10.48550/arXiv.2303.11366](https://doi.org/10.48550/arXiv.2303.11366).

Agrawal, Lakshya A., Aditya Kanade, Navin Goyal, Shuvendu K. Lahiri, and Sriram K. Rajamani. “Guiding Language Models of Code with Global Context Using Monitors.” arXiv, June 19, 2023. [https://doi.org/10.48550/arXiv.2306.10763](https://doi.org/10.48550/arXiv.2306.10763).  
Yang, John, Akshara Prabhakar, Karthik Narasimhan, and Shunyu Yao. “InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback.” arXiv, June 26, 2023. [https://doi.org/10.48550/arXiv.2306.14898](https://doi.org/10.48550/arXiv.2306.14898).  
- [https://intercode-benchmark.github.io/](https://intercode-benchmark.github.io/)

Sanchez, Guillaume, Honglu Fan, Alexander Spangher, Elad Levi, Pawan Sasanka Ammanamanchi, and Stella Biderman. “Stay on Topic with Classifier-Free Guidance.” arXiv, June 30, 2023. [https://doi.org/10.48550/arXiv.2306.17806](https://doi.org/10.48550/arXiv.2306.17806).

### Controlling Output

#### Jsonformer

2023-04-30: " Jsonformer is a wrapper around Hugging Face models that fills in the fixed tokens during the generation process, and only delegates the generation of content tokens to the language model. This makes it more efficient and bulletproof than existing approaches."

- [https://github.com/1rgs/jsonformer](https://github.com/1rgs/jsonformer)

#### Context-Free Grammar Parsing with LLMs

2023-05-14: "Use a context-free grammar and a parser generator to determine valid next tokens for an LLM generation."

- [https://matt-rickard.com/context-free-grammar-parsing-with-llms](https://matt-rickard.com/context-free-grammar-parsing-with-llms)
- [https://github.com/r2d4/parserllm](https://github.com/r2d4/parserllm)
- [https://matt-rickard.com/rellm](https://matt-rickard.com/rellm)

#### Logit Bias

You can use LLMs as an output by using `logit_bias` to control output tokens:

- [https://twitter.com/AAAzzam/status/1669753722828730378](https://twitter.com/AAAzzam/status/1669753722828730378)
- [https://aidungeon.medium.com/controlling-gpt-3-with-logit-bias-55866d593292](https://aidungeon.medium.com/controlling-gpt-3-with-logit-bias-55866d593292)
- [https://help.openai.com/en/articles/5247780-using-logit-bias-to-define-token-probability](https://help.openai.com/en/articles/5247780-using-logit-bias-to-define-token-probability)