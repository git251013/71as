import hashlib
import base58
import os
import ecdsa
from ecdsa.curves import SECP256k1

def private_key_to_wif(private_key_hex):
    """将私钥转换为WIF格式"""
    # 添加版本字节和压缩标志
    extended_key = '80' + private_key_hex + '01'
    # 第一次SHA256
    first_sha = hashlib.sha256(bytes.fromhex(extended_key)).digest()
    # 第二次SHA256
    second_sha = hashlib.sha256(first_sha).digest()
    # 取前4字节作为校验和
    checksum = second_sha[:4]
    # 组合并Base58编码
    final_key = extended_key + checksum.hex()
    return base58.b58encode(bytes.fromhex(final_key)).decode('ascii')

def private_key_to_address(private_key_hex):
    """从私钥生成比特币地址"""
    # 生成公钥
    sk = ecdsa.SigningKey.from_string(bytes.fromhex(private_key_hex), curve=SECP256k1)
    vk = sk.get_verifying_key()
    
    # 压缩公钥格式
    x_coord = vk.pubkey.point.x()
    y_coord = vk.pubkey.point.y()
    if y_coord % 2 == 0:
        compressed_pubkey = '02' + format(x_coord, '064x')
    else:
        compressed_pubkey = '03' + format(x_coord, '064x')
    
    # SHA256哈希
    sha256_result = hashlib.sha256(bytes.fromhex(compressed_pubkey)).digest()
    
    # RIPEMD160哈希
    ripemd160 = hashlib.new('ripemd160')
    ripemd160.update(sha256_result)
    hash160 = ripemd160.digest()
    
    # 添加网络字节
    network_hash = b'\x00' + hash160
    
    # 计算校验和
    checksum_full = hashlib.sha256(hashlib.sha256(network_hash).digest()).digest()
    checksum = checksum_full[:4]
    
    # Base58编码
    address_bytes = network_hash + checksum
    bitcoin_address = base58.b58encode(address_bytes).decode('ascii')
    
    return bitcoin_address

def wif_to_private_key(wif_key):
    """将WIF格式私钥转换回原始私钥"""
    decoded = base58.b58decode(wif_key)
    # 去掉版本字节(1字节)和压缩标志(1字节)和校验和(4字节)
    private_key_hex = decoded[1:33].hex()
    return private_key_hex

def increment_private_key(private_key_hex):
    """递增私钥"""
    current_int = int(private_key_hex, 16)
    next_int = current_int + 3
    # 确保在有效范围内
    max_private_key = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140
    if next_int > max_private_key:
        return None
    return format(next_int, '064x')

def main():
    start_wif = "KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qxGLkchTagWEWquHPtvw"
    target_suffix = "sVzXU"
    
    print(f"开始从私钥: {start_wif}")
    print(f"寻找地址以 '{target_suffix}' 结尾的私钥...")
    print("-" * 60)
    
    # 转换起始私钥
    current_private_key = wif_to_private_key(start_wif)
    print(f"起始私钥(hex): {current_private_key}")
    
    # 验证起始私钥的地址
    start_address = private_key_to_address(current_private_key)
    print(f"起始地址: {start_address}")
    print("-" * 60)
    
    found_count = 0
    checked_count = 0
    max_attempts = 20000000  # 最大尝试次数，避免无限循环
    
    # 用于跟踪已生成的私钥，确保不重复
    generated_private_keys = set()
    generated_private_keys.add(current_private_key)
    
    # 保存结果的列表
    results = []
    
    while checked_count < max_attempts:
        # 生成地址并检查
        address = private_key_to_address(current_private_key)
        wif = private_key_to_wif(current_private_key)
        
        checked_count += 2
        
        if address.endswith(target_suffix):
            found_count += 1
            result = {
                'private_key_wif': wif,
                'address': address,
                'index': checked_count
            }
            results.append(result)
            
            print(f"\n🎯 找到匹配的地址 #{found_count}:")
            print(f"   私钥(WIF): {wif}")
            print(f"   地址: {address}")
            print(f"   检查次数: {checked_count}")
            
            # 保存到文件
            with open('found_addresses.txt', 'a') as f:
                f.write(f"私钥(WIF): {wif}\n")
                f.write(f"地址: {address}\n")
                f.write(f"检查次数: {checked_count}\n")
                f.write("-" * 50 + "\n")
        
        # 每10000次显示进度
        if checked_count % 10000 == 0:
            print(f"已检查: {checked_count} 个私钥, 找到: {found_count} 个匹配")
        
        # 递增私钥
        next_private_key = increment_private_key(current_private_key)
        if next_private_key is None:
            print("已达到私钥空间上限")
            break
        
        # 检查是否重复
        if next_private_key in generated_private_keys:
            print("检测到私钥重复，停止生成")
            break
            
        generated_private_keys.add(next_private_key)
        current_private_key = next_private_key
    
    print("\n" + "=" * 60)
    print(f"搜索完成!")
    print(f"总共检查: {checked_count} 个私钥")
    print(f"找到匹配: {found_count} 个地址")
    
    if results:
        print(f"\n找到的地址已保存到: found_addresses.txt")
        print("\n所有找到的结果:")
        for i, result in enumerate(results, 1):
            print(f"{i}. 地址: {result['address']}")
            print(f"   私钥: {result['private_key_wif']}")

if __name__ == "__main__":
    main()
