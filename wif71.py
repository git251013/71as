import base58
import hashlib
import random

def validate_private_key(private_key_wif):
    """
    验证WIF格式私钥的有效性
    """
    try:
        # Base58解码
        decoded = base58.b58decode(private_key_wif)
        
        # 检查长度（压缩格式应该是38字节）
        if len(decoded) != 38:
            return False, f"长度无效: {len(decoded)} 字节"
        
        # 提取各部分
        version = decoded[0]  # 版本字节
        private_key = decoded[1:33]  # 32字节私钥
        compressed_flag = decoded[33:34]  # 压缩标志
        checksum = decoded[34:38]  # 4字节校验和
        
        # 验证版本字节（比特币主网是0x80）
        if version != 0x80:
            return False, f"版本字节无效: {hex(version)}"
        
        # 验证压缩标志
        if compressed_flag != b'\x01':
            return False, "压缩标志无效"
        
        # 验证私钥范围（应该在1到n-1之间）
        from cryptography.hazmat.primitives.asymmetric import ec
        n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        key_int = int.from_bytes(private_key, 'big')
        if key_int == 0 or key_int >= n:
            return False, "私钥超出有效范围"
        
        # 验证校验和
        payload = decoded[:34]
        computed_checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        
        if computed_checksum != checksum:
            return False, "校验和无效"
        
        return True, "私钥有效"
        
    except Exception as e:
        return False, f"解码错误: {str(e)}"

def generate_valid_private_key():
    """
    生成有效的私钥通过随机补充剩余字符
    """
    base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    partial_key = "KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3y"
    
    attempts = 0
    max_attempts = 100000000
    
    while attempts < max_attempts:
        attempts += 1
        
        # 随机生成19个Base58字符
        random_suffix = ''.join(random.choices(base58_chars, k=19))
        full_key = partial_key + random_suffix
        
        # 验证私钥
        is_valid, message = validate_private_key(full_key)
        
        if is_valid:
            print(f"成功找到有效私钥！尝试次数: {attempts}")
            return full_key, message
        
        if attempts % 10000 == 0:
            print(f"已尝试 {attempts} 次...")
    
    return None, f"在 {max_attempts} 次尝试后未找到有效私钥"

# 主程序
if __name__ == "__main__":
    print("开始搜索有效私钥...")
    print("基础部分: KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3y")
    print("需要补充: 19个字符\n")
    
    result, message = generate_valid_private_key()
    
    if result:
        print(f"\n找到的有效私钥: {result}")
        print(f"验证结果: {message}")
        
        # 解码显示详细信息
        decoded = base58.b58decode(result)
        print(f"\n解码信息:")
        print(f"完整数据 ({len(decoded)}字节): {decoded.hex()}")
        print(f"版本字节: {hex(decoded[0])}")
        print(f"私钥部分: {decoded[1:33].hex()}")
        print(f"压缩标志: {decoded[33:34].hex()}")
        print(f"校验和: {decoded[34:38].hex()}")
    else:
        print(f"\n{message}")
