import hashlib
import base58
import ecdsa
import os

def wif_to_private_key(wif):
    """将WIF格式私钥转换为原始私钥字节"""
    # 解码Base58Check
    decoded = base58.b58decode_check(wif)
    
    # 对于主网WIF压缩格式，去掉前缀0x80和后缀0x01
    if len(decoded) == 38:  # 压缩格式
        return decoded[1:33]  # 去掉前缀0x80和压缩标志0x01
    else:  # 非压缩格式
        return decoded[1:33]

def private_key_to_wif(private_key_bytes, compressed=True):
    """将私钥字节转换为WIF格式"""
    # 添加主网前缀0x80
    extended_key = b'\x80' + private_key_bytes
    
    if compressed:
        extended_key += b'\x01'  # 添加压缩标志
    
    # 计算校验和
    first_sha256 = hashlib.sha256(extended_key).digest()
    second_sha256 = hashlib.sha256(first_sha256).digest()
    checksum = second_sha256[:4]
    
    # 组合并编码为Base58
    final_key = extended_key + checksum
    return base58.b58encode(final_key).decode('ascii')

def private_key_to_address(private_key_bytes):
    """从私钥生成比特币地址"""
    # 生成公钥
    sk = ecdsa.SigningKey.from_string(private_key_bytes, curve=ecdsa.SECP256k1)
    vk = sk.get_verifying_key()
    
    # 压缩公钥格式
    x = vk.pubkey.point.x()
    y = vk.pubkey.point.y()
    if y % 2 == 0:
        compressed_pubkey = b'\x02' + x.to_bytes(32, 'big')
    else:
        compressed_pubkey = b'\x03' + x.to_bytes(32, 'big')
    
    # 生成地址
    sha256_hash = hashlib.sha256(compressed_pubkey).digest()
    ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
    
    # 添加主网前缀0x00
    network_byte = b'\x00' + ripemd160_hash
    
    # 计算校验和
    first_sha256 = hashlib.sha256(network_byte).digest()
    second_sha256 = hashlib.sha256(first_sha256).digest()
    checksum = second_sha256[:4]
    
    # 组合并编码为Base58
    final_address = network_byte + checksum
    return base58.b58encode(final_address).decode('ascii')

def increment_private_key(private_key_bytes):
    """将私钥加1"""
    private_key_int = int.from_bytes(private_key_bytes, 'big')
    new_private_key_int = (private_key_int + 3) % (0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141)
    return new_private_key_int.to_bytes(32, 'big')

def main():
    # 起始WIF私钥
    start_wif = "KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qxGLkchTagWEWquHPtvw"
    target_address = "1PWo3Je"
    
    print(f"起始私钥: {start_wif}")
    print(f"目标地址: {target_address}")
    print("开始搜索...")
    
    # 转换起始私钥
    current_private_key = wif_to_private_key(start_wif)
    current_wif = start_wif
    
    found = False
    attempts = 0
    max_attempts = 30000000  # 安全限制，避免无限循环
    
    while not found and attempts < max_attempts:
        # 生成地址
        address = private_key_to_address(current_private_key)
        
        # 检查是否匹配目标地址
        if address == target_address:
            print(f"\n🎉 找到匹配的地址!")
            print(f"私钥 (WIF): {current_wif}")
            print(f"地址: {address}")
            found = True
            break
        
        # 递增私钥
        current_private_key = increment_private_key(current_private_key)
        current_wif = private_key_to_wif(current_private_key)
        attempts += 2
        
        # 每10000次显示进度
        if attempts % 10000 == 0:
            print(f"已尝试 {attempts} 次... 当前地址: {address}")
    
    if not found:
        print(f"\n在 {attempts} 次尝试后未找到匹配的地址")
    
    return found, current_wif if found else None

if __name__ == "__main__":
    success, found_wif = main()
    
    if success:
        print(f"\n匹配的私钥已找到并显示在上面")
        print("注意：请妥善保管私钥，确保安全")
    else:
        print("未找到匹配的地址")
