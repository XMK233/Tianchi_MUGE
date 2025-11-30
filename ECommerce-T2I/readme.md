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
