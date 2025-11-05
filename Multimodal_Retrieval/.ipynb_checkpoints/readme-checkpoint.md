电商文本到图像生成
1. 任务描述
文本到图像生成是一项具有挑战性的任务，它需要图像生成以及对跨模态理解的能力。任务目标是根据一段文本描述，生成符合相应描述的图像，同时要求图像清晰且逼真。电商领域有着众多的商品图片，将文本到图像生成技术应用于电商领域，对于商品上新、设计、分发，减少商家运营成本，提高用户体验有着重要的意义。

2. 任务说明
本项任务涵盖了服装、饰品、化妆品内的多个商品类目，对于一件商品，我们会给出它的商品展示图片以及相应的商品描述作为训练数据，其中每张图片均以base64编码表示，参赛者需要根据商品描述生成对应的商品图片。标注数据的示例如下：

文本数据，tsv格式（图片id，\t，商品描述）
8cf9ceb2a031d5a7fc88482b8a2b2fa6	男女童纯棉短裤2021夏季新款宝宝帅气五分裤洋气裤子运动裤
图片数据，tsv格式（图片id，\t，商品图片内容) (base64编码）
8cf9ceb2a031d5a7fc88482b8a2b2fa6	iVBORw0KGgoAAAANSUhEUgAA...
解码图片base64编码，参照代码：
import base64
from io import BytesIO
from PIL import Image
img = Image.open(BytesIO(base64.urlsafe_b64decode(image_base64)))
编码图片base64编码（生成测试数据答案需用到），参照代码：
import base64
from io import BytesIO
from PIL import Image

img = Image.open(fn)
img_buffer = BytesIO()
img.save(img_buffer, format=img.format)
byte_data = img_buffer.getvalue()
base64_str = base64.b64encode(byte_data) # bytes
3. 评测指标
评测指标分为三项：

FID：使用Inception v3网络抽取的图像特征，比较生成图像和真实图像之间的Fr´echet 距离，FID越小代表距离约接近。
IS：使用Inception v3网络计算条件类别分布和边缘类别分布的KL散度，IS越大代表生成图像的多样性和真实性越高。
R-precision：比较文本描述和生成图像之间的相似度
其中FID为主指标。

4. 评测数据
本评测开放训练集数据90000张图片，验证集数据5000张图片，测试集5000张图片，其中每张图片对应一条描述，训练集、验证集、测试集描述均不重合。

数据集名称为：ECommerce-T2I (E-commerce Text to Image Dataset)

数据集下载文件为：ECommerce-T2I.zip，包括：

T2I_train.txt.tsv: 训练集对应的文本数据
T2I_train.img.tsv：训练集对应的图片数据（base64编码）
T2I_val.txt.tsv: 验证集对应的文本数据
T2I_val.img.tsv：验证集对应的图片数据（base64编码）
T2I_test.txt.tsv：测试集对应的文本数据
T2I_test.tsv: 需选手提交的文件，该文件第一列是img_id, 选手需要为每条img_id补充生成图片的base64编码，用tab隔开。
example_gold.tsv：标准答案示例
example_pred.tsv：提交结果示例
README.txt：说明文件
5. 数据集信息
数据集提供方
阿里巴巴达摩院智能计算实验室

6. Github
https://github.com/MUGE-2021/image-generation-baseline

电商图文检索
1. 任务描述
电商图文检索任务要求模型根据自然语言形式的检索query，从给定的商品图片池中检索出相关图片，衡量模型多模态理解与匹配的能力。在实际的电商业务中，多模态检索扮演重要的角色，是电商场景满足用户需求、促成点击交易不可缺少的一环。为了更多侧重多模态匹配，我们此次不公开商品的标题以及其他信息，要求模型仅基于商品图片进行检索召回。

2. 任务说明
本次任务使用的电商检索数据涵盖服装、家居、电子等多个领域，由商品图片和搜索query两部分构成，划分为训练集、验证集和测试集。对于商品图片，我们将原图统一缩放为224*224大小，以base64编码格式表示。我们为训练集、验证集和测试集分别提供一个商品图片集合，具体细节如下：

商品图片数据为tsv格式（商品图片id，'\t'，商品图片内容) (base64编码），训练集、验证集和测试集的商品图片及其id没有交集
1000002	/9j/4AAQSkZJ...YQj7314oA//2Q==
1000016	/9j/4AAQSkZJ...SSkHcOnegD/2Q==
1000033	/9j/4AAQSkZJ...FhRRRWx4p//2Q==
解码图片base64编码，可参照代码：
import base64
from io import BytesIO
from PIL import Image
img = Image.open(BytesIO(base64.urlsafe_b64decode(image_base64)))
对于搜索query，训练集和验证集中给出了query的id、文本内容及其对应的商品图片id（分别来自的训练集和验证集各自的商品集合）。测试集则仅给出query的id和文本，需要模型从测试集商品集合中预测相关商品。训练集query一般对应1-2个商品图片，验证集和测试集query平均对应6个商品图片。训练集、验证集和测试集之间query没有交集。query数据具体格式如下：

训练集&验证集query：jsonl格式，包含ground truth商品id
{"query_id": 8426, "query_text": "胖妹妹松紧腰长裤", "item_ids": [42967]}
{"query_id": 8427, "query_text": "大码长款棉麻女衬衫", "item_ids": [63397]}
{"query_id": 8428, "query_text": "高级感托特包斜挎", "item_ids": [1076345, 517602]}
测试集query：jsonl格式，不含ground truth
{"query_id": 352962, "query_text": "厚底系带帆布鞋女"}
{"query_id": 352976, "query_text": "日式粗陶咖啡杯"}
{"query_id": 353590, "query_text": "豹纹雪纺半身"}
3. 评测指标
我们要求选手提交jsonl格式的测试集预测结果，格式与前述验证集query文件相同，item_ids字段按top1-top10顺序给出模型预测的top10结果。参考MSCOCO、Flickr30K等英文检索数据集，我们也同样用Recall@1/5/10作为评测指标。该指标统计预测结果topk(k=1/5/10)中含有至少1个ground truth商品图片的query数量，除以测试集query总数。我们以Recall@1/5/10的平均值(MeanRecall)作为该任务主指标。

4. 评测数据
本次评测训练集包含25w搜索query-相关商品对，覆盖12.9w商品图片。验证集和测试集各自包含5k搜索query，要求模型从各自的3w商品图片候选池中进行检索。

数据集名称为：Multimodal Retrieval Dataset

数据集放在文件夹：/mnt/d/forCoding_code/Tianchi_MUGE/Multimodal_Retrieval/ 下，文件的内容和含义：

MR_train_imgs.tsv：训练集图片集合（base64编码）
MR_train_queries.jsonl：训练集搜索query及对应商品id
MR_valid_imgs.tsv：验证集图片集合（base64编码）
MR_valid_queries.jsonl：验证集搜索query及对应商品id
MR_test_imgs.tsv：测试集图片集合（base64编码）
MR_test_queries.jsonl：测试集搜索query，需预测的文件，选手需要补充"item_id"字段，为list类型。
example_pred.jsonl：测试集提交结果示例
README.txt: readme.

