## 将当前目录下的文件夹名字，比如4.9、4.10这种，做进python列表中，然后
## 你从这些文件夹里的某个同名文件里面，获取
## 图片（可能是base64编码），然后并列放在一起，
## 每个图片要有一个标题，标题就是文件夹的名字。
## 我这么做是为了横向对比不同的方案生成的图片的效果好坏。
## 

import os
import re

# 目标文件名，可以根据需要修改，比如 page_40.html 或 page_6.html
# 这里选择 page_6.html 作为示例，因为它是多个实验中都存在的较新文件
TARGET_FILENAME = "page_6.html"
OUTPUT_FILE = "comparison_result.html"

def get_directories(root_dir):
    """获取当前目录下以 4. 开头的文件夹"""
    dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("4.")]
    # 按名称排序
    dirs.sort()
    return dirs

def extract_images_and_text(filepath):
    """从html文件中提取生成的图片和对应的文本"""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 分割每个图片块
    # 假设每个块以 <div style="display:flex;gap:24px;margin-bottom:16px;"> 开始
    blocks = content.split('<div style="display:flex;gap:24px;margin-bottom:16px;">')
    
    results = []
    
    for block in blocks[1:]: # 跳过第一个（通常是header前的部分）
        # 提取文本
        # <div>text: ...</div>
        text_match = re.search(r'<div>text: (.*?)</div>', block)
        text = text_match.group(1) if text_match else "Unknown"
        
        # 提取生成图片
        # 寻找 <div>生成图片</div> 后的 <img src="...">
        # 注意：使用 re.DOTALL 让 . 匹配换行符
        gen_img_match = re.search(r'<div>生成图片</div>\s*<img src="([^"]+)"', block, re.DOTALL)
        img_src = gen_img_match.group(1) if gen_img_match else None
        
        results.append({'text': text, 'img': img_src})
        
    return results

def generate_html(data, dirs, output_file):
    """生成对比的 HTML 文件"""
    html = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>横向对比生成图片</title>
        <style>
            table { border-collapse: collapse; width: 100%; table-layout: fixed;}
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; vertical-align: top; word-wrap: break-word;}
            img { max-width: 100%; height: auto; }
            th { background-color: #f2f2f2; }
            .prompt-col { width: 150px; font-weight: bold;}
        </style>
    </head>
    <body>
        <h1>横向对比生成图片 (来源文件: """ + TARGET_FILENAME + """)</h1>
        <table>
            <thead>
                <tr>
                    <th class="prompt-col">Prompt / Text</th>
    """
    
    # 表头：文件夹名
    for d in dirs:
        html += f"<th>{d}</th>"
    html += "</tr></thead><tbody>"
    
    # 确定行数（取最大行数）
    max_rows = 0
    for d in dirs:
        if data[d]:
            max_rows = max(max_rows, len(data[d]))
            
    for i in range(max_rows):
        html += "<tr>"
        
        # 获取这一行的文本描述 (尝试从第一个有数据的列获取)
        row_text = ""
        for d in dirs:
            if data[d] and i < len(data[d]):
                row_text = data[d][i]['text']
                break
        
        html += f"<td>{row_text}</td>"
        
        # 每一列的图片
        for d in dirs:
            if data[d] and i < len(data[d]) and data[d][i]['img']:
                html += f'<td><img src="{data[d][i]["img"]}"></td>'
            else:
                html += "<td>N/A</td>"
        
        html += "</tr>"
        
    html += "</tbody></table></body></html>"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"对比文件已生成: {output_file}")

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = get_directories(root_dir)
    
    data = {}
    
    print(f"正在扫描文件夹 (目标文件: {TARGET_FILENAME})...")
    
    for d in dirs:
        filepath = os.path.join(root_dir, d, TARGET_FILENAME)
        print(f"处理: {d} ... ", end="")
        results = extract_images_and_text(filepath)
        if results:
            data[d] = results
            print(f"找到 {len(results)} 张图片")
        else:
            data[d] = []
            print("文件不存在或无内容")
            
    output_path = os.path.join(root_dir, OUTPUT_FILE)
    generate_html(data, dirs, output_path)

if __name__ == "__main__":
    main()
