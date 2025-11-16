# 3.1

## 实现模型微调策略，适应电商领域数据

* step_3_1-1_乾-基线：~~【折桂】Recall@1=0.0587, Recall@5=0.1979, Recall@10=0.3071, MeanRecall=0.1879 (N=5008)~~ **启发**：`换好的预训练模型，有用。`
* step_3_1-2_坤-使用微调后模型：基于1，但是用预训练过的模型来接着训。没有提升。
* step_3_1-4_蒙：
    * step_3_1-4_蒙_cp3-基于3_更换池化为attentive：~~【折桂】Recall@1=0.0733, Recall@5=0.2113, Recall@10=0.3091, MeanRecall=0.1979 (N=5008)~~ 些微有点提高。
    * step_3_1-4_蒙_cp1-基于3_更换池化为cls：表现没有很大提高。
* step_3_1-5_需：
    * step_3_1-5_需_cp1-基于4_cp2-解冻4层：~~【折桂】Recall@1=0.0681, Recall@5=0.2272, Recall@10=0.3357, MeanRecall=0.2103 (N=5008)~~
    * step_3_1-5_需_cp2-基于4_cp2-解冻8层：~~【折桂】Recall@1=0.0749, Recall@5=0.2410, Recall@10=0.3506, MeanRecall=0.2222 (N=5008)~~ **启发**：`解冻得越多，越好。后续可以考虑进一步解冻，乃至解冻embedding等层。`
* step_3_1-6_讼：
    * step_3_1-6_讼_cp2-基于5_cp2-换图像模型2：~~【折桂】Recall@1=0.0801, Recall@5=0.2438, Recall@10=0.3580, MeanRecall=0.2273 (N=5008)~~ 只寻3轮，表现就能提升。后续或可考虑增加训练轮数到5轮。Recall@1=0.0779, Recall@5=0.2392, Recall@10=0.3526, MeanRecall=0.2232 (N=5008) 没有有效用的提升。
    * step_3_1-6_讼_cp1-基于5_cp2-换图像模型1：~~基于convnext-tiny，表现反为不美，后续考虑缩减训练轮数再试。还是不好。会不会还是过拟合了？~~ 就是写错了。加载timm模型的时候，还是得设置pretrain为True。使用了convnext之后表现暴涨。**Recall@1=0.1326, Recall@5=0.3345, Recall@10=0.4641, MeanRecall=0.3104 (N=5008)**
    * step_3_1-6_讼_cp3-基于5_cp2-换图像模型3：尝试用vit作为骨干模型。一般吧，后来查了一下，好像convnext就是为了突破vit而生的。Recall@1=0.1012, Recall@5=0.2847, Recall@10=0.4056, MeanRecall=0.2638 (N=5008)
* step_3_1-7_师：gap还是fc。gap就是原来的样子，fc就是要做一些改动的。
    * step_3_1-7_师_cp1-基于6_cp2-fc前特征也就是gap：说白了就是对照组，没有实质改动。Recall@1=0.0829, Recall@5=0.2384, Recall@10=0.3492, MeanRecall=0.2235 (N=5008)
    * step_3_1-7_师_cp2-基于6_cp2-伪fc：~~【折桂】Recall@1=0.0827, Recall@5=0.2458, Recall@10=0.3544, MeanRecall=0.2276 (N=5008)~~
    * step_3_1-7_师_cp3-基于6_cp2-真fc需要改num_classes=1000: Recall@1=0.0663, Recall@5=0.2171, Recall@10=0.3173, MeanRecall=0.2002 (N=5008) 优势不大。
* step_3_1-8_比：
    * step_3_1-8_比-基于7_cp2-全解冻并结合LLRD：~~【折桂】Recall@1=0.0861, Recall@5=0.2500, Recall@10=0.3654, MeanRecall=0.2338 (N=5008)~~ **启发**: `解冻越多还是越好啊。。。干脆横下一条心，我们把文本模型也直接全解冻了试试？因为文本模型只解冻了8层，估计还远没到顶，可以先把所有的层都解冻试试、不要动embedding层`
* step_3_1-9_小畜：
    * step_3_1-9_小畜_cp1-基于8-新的模态融合【后融合】：原本就是后融合。~~_【折桂】_ Recall@1=0.0887, Recall@5=0.2512, Recall@10=0.3744, MeanRecall=0.2381 (N=5008)~~
    * step_3_1-9_小畜_cp1-基于8-新的模态融合【早融合】：~~没效果。感觉好像根本没有训练到。不知道是不是代码写错了。~~ 修正了一下代码，表现就正常了。Recall@1=0.0927, Recall@5=0.2540, Recall@10=0.3632, MeanRecall=0.2366 (N=5008) **启发**：会不会之前convnext之类的表现稀烂，也是哪里写错了导致的？
* step_3_1-10_履：
    * step_3_1-10_履-基于9_cp1-txt和img共享投影头：效果变差了。不要了。
* step_3_1-11_泰：
    * step_3_1-11_泰-基于9_cp1-相似度函数改为dot：不好。Recall@1=0.0525, Recall@5=0.1601, Recall@10=0.2546, MeanRecall=0.1558 (N=5008)
* step_3_1-12_否_cp1：
    * step_3_1-12_否_cp1-基于9_cp1-图文统一可学温度：Recall@1=0.0829, Recall@5=0.2500, Recall@10=0.3684, MeanRecall=0.2338 (N=5008)
    * step_3_1-12_否_cp2-基于9_cp1-图文不同可学温度：Recall@1=0.0831, Recall@5=0.2456, Recall@10=0.3572, MeanRecall=0.2286 (N=5008) 这俩都没有显著改变模型表现。带来的影响感觉挺随机的。
* step_3_1-13_同人_cp1：
    * step_3_1-13_同人_cp1：step_3_1-13_同人_cp1-基于9_cp1-换convnext试试：~~_【折桂】_ Recall@1=0.1184, Recall@5=0.3181, Recall@10=0.4369, MeanRecall=0.2911 (N=5008)~~
* step_3_1-14_大有：
    * step_3_1-14_大有-基于6_cp1-cnvnxt基础上多解冻、schd、llrd：Recall@1=0.1224, Recall@5=0.3193, Recall@10=0.4367, MeanRecall=0.2928 (N=5008) 没有更好。
* step_3_1-15_谦：
    * step_3_1-15_谦_cp1-基于6_cp1-难例挖掘：一开始是 Recall@1=0.1288, Recall@5=0.3403, Recall@10=0.4617, MeanRecall=0.3102 (N=5008) 后来想再调一下，就成了这德行 Recall@1=0.0373, Recall@5=0.1296, Recall@10=0.2165, MeanRecall=0.1278 (N=5008)
    * step_3_1-15_谦_cp2-基于6_cp1-记忆队列：训练3轮，效果不好。Recall@1=0.0333, Recall@5=0.1308, Recall@10=0.2131, MeanRecall=0.1257 (N=5008) 增多几个epoch试试？没卵用 Recall@1=0.0310, Recall@5=0.1184, Recall@10=0.2065, MeanRecall=0.1186 (N=5008)
* step_3_1-16_豫：
    * step_3_1-16_豫_cp3-基于6_cp1-图像和文本增强：图有三种增强模式【"none", "jitter_flip", "grayscale_blur"】，文有三种增强模式【"none", "word_dropout", "random_swap"】。考虑到有数据增强，所以都提高了训练轮次数到6轮。
        * grayscale_blur_random_swap：Recall@1=0.1328, Recall@5=0.3407, Recall@10=0.4756, MeanRecall=0.3164 (N=5008)
        * grayscale_blur_word_dropout：Recall@1=0.1256, Recall@5=0.3353, Recall@10=0.4611, MeanRecall=0.3073 (N=5008)
        * grayscale_blur_none：Recall@1=0.1286, Recall@5=0.3397, Recall@10=0.4617, MeanRecall=0.3100 (N=5008)
        * jitter_flip_word_dropout：Recall@1=0.1264, Recall@5=0.3325, Recall@10=0.4591, MeanRecall=0.3060 (N=5008)
        * jitter_flip_none：Recall@1=0.1248, Recall@5=0.3321, Recall@10=0.4629, MeanRecall=0.3066 (N=5008)
        * ？？？
        * none_random_swap：Recall@1=0.1300, Recall@5=0.3466, Recall@10=0.4740, MeanRecall=0.3169 (N=5008)
        * none_word_dropout：_【折桂】_ Recall@1=0.1360, Recall@5=0.3460, Recall@10=0.4790, MeanRecall=0.3204 (N=5008) **启发**：感觉增强文本比增强图像更有收效。
        * none_none：Recall@1=0.1384, Recall@5=0.3464, Recall@10=0.4746, MeanRecall=0.3198 (N=5008)

# 2.3

## 基线

表现很普通。

## 解冻轻量微调

* step_2_3-2_坤-解冻轻量微调-v2_amp_faiss：表现有明显提升。

## mean_pooling

* step_2_3-3_屯-mean_pooling：训练5轮 ~~【折桂】Recall@1=0.0571, Recall@5=0.1905, Recall@10=0.2927, MeanRecall=0.1801 (N=5008)~~。不确定这里之所以有比较明显的提升是因为随机性还是其他原因。
* step_2_3-3_屯-mean_pooling-v2：
    * 把projector改复杂了一些，训练10轮，才能让loss降到之前的水准。表现反而显著下降了。
* step_2_3-3_屯-mean_pooling-v2_cp1：
    * 恢复了简单的projector，代码更规整。没有实质改进。
    * 解冻2层训练5轮，表现还好点。解冻多了训练轮数多了反而不好。
* step_2_3-3_屯-mean_pooling-v3：增加了学习率schedule。表现似乎倒退了。
* 

## 可学习温度

总体没用。倒回去。

* step_2_3-4_蒙-两层MLP_可学习温度-v4：较为完整的代码。

## 使用clip

* step_2_3-5_需-CLIP双编码器_最小改动：简单地使用clip。效果很差。