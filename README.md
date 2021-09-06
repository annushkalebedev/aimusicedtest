## aimusic 人工智能音乐教育系统

### 运行

安装依赖
```pip install requirements.txt```

运行
```streamlit run app.py```

浏览器打开 ```localhost:8001```

### 文件内容

```
./aimusic/
│
├── pages/
│   ├── melody.py        旋律页面
│   ├── development.py   发展旋律页面
│   ├── structure.py     结构页面
│   ├── harmony.py       和弦页面
│   └── arrangement.py   后期页面
│
├── README.md
├── app.py                主程序
├── evolution.py          遗传算法
├── markov.py             马尔科夫链
├── RNN.py                RNN
├── params.py             全局参数
├── utils.py              辅助函数
├── preprocess.py         数据集预处理
│
├── assets/                生成产物，提取的数据
│   ├── token_lists_*.txt  数据集中提取的旋律
│   ├── *_pitch_markov.csv 马尔科夫矩阵
│   ├── melody_*.mid       生成导出的midi
│   ├── melody_*.wav       midi合成的wav
│   ├── melody_*.png       midi生成的卷轴谱
│   └── rnnmodel.pt        rnn模型
└──
```

### Streamlit框架 和 session_state

本应用使用了Streamlit框架，一个针对机器学习应用的网页交互工具。它完全基于python，其中UI组件(按钮，表单)和数据计算都是用python实现。运行时，```app.py``` 是程序的入口，其中引用了五个页面(见```/pages```)进行不同的任务。


#### State 

作为一个交互型应用，记录用户输入以及应用状态是必不可少的。音乐参数(调号，节拍)，生成产物(生成的旋律)等都存在于整个应用的全局状态(State)，可以通过```st.session_state```调取。初始化这些状态在```utils.py```中。


### 算法

生成乐曲主要使用到算法包括：
* 1. 主旋律生成 - 马尔科夫链
* 2. 副旋律生成 - 遗传算法
* 3. 曲式结构 - 生成语法
* 4. 和声 - 序列模型（HMM, RNN）

部分代码参考：
[序列模型](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)
[遗传算法](https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6)

### 数据

使用的三个开源数据集(未上传到仓库中)：[Wikifonia](http://www.synthzone.com/forum/ubbthreads.php/topics/384909/Download_for_Wikifonia_all_6_6), [Hooktheory](https://github.com/owencm/hooktheory-data) 和 [POP909](https://github.com/music-x-lab/POP909-Dataset). 在主旋律生成中，使用了三个数据集的旋律信息得到马尔科夫矩阵；在配和声算法中，使用了Hooktheory的旋律和和声数据。

在预处理 ```preprocess.py```中，有从raw数据提取旋律/和声序列的函数。


### 多媒体渲染

该应用中的多媒体(音频，图片)都是预先生成到本地```/assets```，然后再渲染到页面上来。用户可以在界面上下载它们。

### 其他

应用的配置在```.streamlit/config.toml```里面，比如配色。
