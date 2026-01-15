import base64
from io import BytesIO
from PIL import Image

def decode_base64_image(image_base64):
    """
    解码base64编码的图片数据
    来自readme.md第13-16行的方法
    """
    img = Image.open(BytesIO(base64.urlsafe_b64decode(image_base64)))
    return img

def encode_image_to_base64(image_path):
    """
    将图片编码为base64格式
    """
    img = Image.open(image_path)
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str

def read_image_chunk(image_file_path, start_line, chunk_size):
    """
    从图片文件中读取指定范围的行
    
    参数:
        image_file_path: 图片文件路径
        start_line: 起始行（从0开始）
        chunk_size: 要读取的行数
    
    返回:
        list: 包含(img_id, image_base64)的元组列表
    """
    result = []
    current_line = 0
    
    try:
        with open(image_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if current_line >= start_line + chunk_size:
                    break
                if current_line >= start_line:
                    # 解析行数据
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        img_id, image_base64 = parts
                        result.append((img_id, image_base64))
                current_line += 1
    except FileNotFoundError:
        print(f"图片文件 {image_file_path} 不存在")
    except Exception as e:
        print(f"读取图片文件时出错: {str(e)}")
    
    return result

def find_texts_by_ids(text_file_path, target_img_ids):
    """
    根据图片ID列表从文本文件中查找对应的描述
    
    参数:
        text_file_path: 文本文件路径
        target_img_ids: 要查找的img_id集合
    
    返回:
        dict: {img_id: 商品描述}的字典
    """
    text_dict = {}
    target_set = set(target_img_ids)
    
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_id, description = parts
                    if img_id in target_set:
                        text_dict[img_id] = description
                        # 如果所有目标ID都找到了，可以提前退出
                        if len(text_dict) == len(target_set):
                            break
    except FileNotFoundError:
        print(f"文本文件 {text_file_path} 不存在")
    except Exception as e:
        print(f"读取文本文件时出错: {str(e)}")
    
    return text_dict

class ChunkLoader:
    def __init__(self, image_file_path, text_file_path, chunk_size=1000):
        """
        初始化分片加载器
        
        参数:
            image_file_path: 图片文件路径
            text_file_path: 文本文件路径
            chunk_size: 默认分片大小
        """
        self.image_file_path = image_file_path
        self.text_file_path = text_file_path
        self.chunk_size = chunk_size
        
    def get_chunk(self, start_line=None, chunk_size=None):
        """
        获取指定范围的数据分片
        
        参数:
            start_line: 起始行（从0开始），如果为None则随机选择
            chunk_size: 分片大小，如果为None则使用默认值
        
        返回:
            list: 包含(img_id, image_base64, description)的元组列表
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        # 如果未指定起始行，则随机选择（可选实现）
        if start_line is None:
            # 可以通过文件行数随机选择起始行
            # 这里简化实现，默认从0开始
            start_line = 0
        
        # 1. 读取图片数据分片
        image_chunk = read_image_chunk(self.image_file_path, start_line, chunk_size)
        
        if not image_chunk:
            print(f"未读取到图片数据，起始行: {start_line}, 分片大小: {chunk_size}")
            return []
        
        # 2. 提取图片ID
        img_ids = [img_id for img_id, _ in image_chunk]
        
        # 3. 查找对应的文本描述
        text_dict = find_texts_by_ids(self.text_file_path, img_ids)
        
        if not text_dict:
            print(f"未找到任何文本描述，图片ID数量: {len(img_ids)}")
            return []
        
        # 4. 组装数据
        result = []
        for img_id, image_base64 in image_chunk:
            if img_id in text_dict:
                result.append((img_id, image_base64, text_dict[img_id]))
        
        return result
    
    def get_total_lines(self, file_path):
        """
        获取文件总行数（用于确定分片范围）
        
        参数:
            file_path: 文件路径
        
        返回:
            int: 文件总行数
        """
        import subprocess
        try:
            result = subprocess.run(['wc', '-l', file_path], 
                                  capture_output=True, text=True)
            return int(result.stdout.split()[0])
        except Exception as e:
            print(f"获取文件总行数时出错: {str(e)}")
            return 0

if __name__ == "__main__":
    # 测试代码
    print("开始测试分片加载器...")
    
    # 使用实际的数据文件路径
    image_file_path = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.img.tsv"
    text_file_path = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.text.tsv"
    
    # 初始化加载器
    loader = ChunkLoader(image_file_path, text_file_path, chunk_size=10)
    
    # 加载10行数据
    print("\n正在加载10行数据...")
    chunk_data = loader.get_chunk(start_line=0, chunk_size=10)
    
    # 打印结果
    print(f"\n成功加载 {len(chunk_data)} 行数据")
    
    if chunk_data:
        for i, (img_id, image_base64, description) in enumerate(chunk_data):
            print(f"\n第 {i+1} 条数据:")
            print(f"图片ID: {img_id}")
            print(f"描述: {description[:50]}...")
            print(f"图片Base64长度: {len(image_base64)}")
            
            # 测试解码图片
            try:
                img = decode_base64_image(image_base64)
                print(f"图片尺寸: {img.size}")
                print(f"图片模式: {img.mode}")
            except Exception as e:
                print(f"解码图片时出错: {str(e)}")
    
    print("\n测试完成!")
