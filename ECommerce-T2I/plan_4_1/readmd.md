如果训练20epoch的话，基本上15号一骑绝尘，唯一一个能看出人形的。

后续我们尝试再训练20轮吧。





==========

本文件夹，我们尝试换当时的汉服数据集，来钻研一下算法的各个部分都有什么作用。其实就是消融实验，之所以用汉服数据，就是为了看着更有趣一点，迭代快一点。
* x-处理汉服图片 这个文件将文件拷贝。
* process_selected 这个文件能够获取图片里的文本，放到 dataset_processed.txt 中。
* process_text_extraction 这个文件能够简化 dataset_processed.txt 中的文本，便于后续训练。
* prepare_hanfu_dataset.py 这个文件将 dataset_processed.txt 中的文本，和汉服图片，转为适合本次任务的tsv形式，以便无缝衔接代码。

