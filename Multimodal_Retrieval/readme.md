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

数据集放在文件夹：`/mnt/d/forCoding_data/Tianchi_MUGE/originalData/Multimodal_Retrieval/` 下，文件的内容和含义：

MR_train_imgs.tsv：训练集图片集合（base64编码）
MR_train_queries.jsonl：训练集搜索query及对应商品id
MR_valid_imgs.tsv：验证集图片集合（base64编码）
MR_valid_queries.jsonl：验证集搜索query及对应商品id
MR_test_imgs.tsv：测试集图片集合（base64编码）
MR_test_queries.jsonl：测试集搜索query，需预测的文件，选手需要补充"item_id"字段，为list类型。
example_pred.jsonl：测试集提交结果示例
README.txt: readme.

