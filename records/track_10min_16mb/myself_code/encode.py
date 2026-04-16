import lzma
import base64


def transform_code(input_file, output_file):
    # 1. 读取源代码
    with open(input_file, 'rb') as f:
        original_data = f.read()

    # 2. 使用 LZMA 算法进行极限压缩
    # preset=9 是最高压缩率
    compressed_data = lzma.compress(original_data, preset=9)

    # 3. 使用 Base85 编码转为文本
    b85_encoded = base64.b85encode(compressed_data).decode('utf-8')

    # 4. 生成目标代码模板
    # 这段代码模拟了你给出的 L.decompress 形式
    final_template = (
        f"import lzma as L, base64 as B\n"
        f"exec(L.decompress(B.b85decode('{b85_encoded}')))"
    )

    with open(output_file, 'w') as f:
        f.write(final_template)

    print(f"转换成功！输出文件: {output_file}")
    print(f"压缩比: {len(final_template) / len(original_data) * 100:.2f}%")

# 使用示例
transform_code('train_gpt.py', 'train_gpt_encode.py')