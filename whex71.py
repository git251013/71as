import hashlib
import base58
import ecdsa
from ecdsa.curves import SECP256k1

def private_key_to_wif(private_key_hex, compressed=True):
    """将16进制私钥转换为WIF格式"""
    private_key_bytes = bytes.fromhex(private_key_hex)
    
    # 添加主网版本字节 (0x80)
    extended_key = b'\x80' + private_key_bytes
    
    if compressed:
        extended_key += b'\x01'
    
    # 双重SHA256哈希
    first_hash = hashlib.sha256(extended_key).digest()
    second_hash = hashlib.sha256(first_hash).digest()
    
    # 添加校验和 (前4字节)
    checksum = second_hash[:4]
    final_key = extended_key + checksum
    
    # Base58编码
    wif = base58.b58encode(final_key)
    return wif.decode('utf-8')

def private_key_to_address(private_key_hex, compressed=True):
    """从私钥生成比特币地址"""
    private_key_bytes = bytes.fromhex(private_key_hex)
    
    # 生成公钥
    sk = ecdsa.SigningKey.from_string(private_key_bytes, curve=SECP256k1)
    vk = sk.get_verifying_key()
    
    if compressed:
        # 压缩公钥
        x = vk.pubkey.point.x()
        y = vk.pubkey.point.y()
        if y % 2 == 0:
            public_key = b'\x02' + x.to_bytes(32, 'big')
        else:
            public_key = b'\x03' + x.to_bytes(32, 'big')
    else:
        # 非压缩公钥
        public_key = b'\x04' + vk.pubkey.point.x().to_bytes(32, 'big') + vk.pubkey.point.y().to_bytes(32, 'big')
    
    # SHA256哈希
    sha256_hash = hashlib.sha256(public_key).digest()
    
    # RIPEMD160哈希
    ripemd160 = hashlib.new('ripemd160')
    ripemd160.update(sha256_hash)
    ripemd160_hash = ripemd160.digest()
    
    # 添加网络字节 (0x00 主网)
    network_byte = b'\x00' + ripemd160_hash
    
    # 双重SHA256哈希计算校验和
    first_checksum = hashlib.sha256(network_byte).digest()
    second_checksum = hashlib.sha256(first_checksum).digest()
    checksum = second_checksum[:4]
    
    # 最终地址字节
    address_bytes = network_byte + checksum
    
    # Base58编码
    address = base58.b58encode(address_bytes)
    return address.decode('utf-8')

def find_specific_address(start_value, count, target_address):
    """从起始值开始生成私钥，寻找特定地址"""
    print(f"从 {hex(start_value)} 开始搜索 {count} 个私钥，寻找地址: {target_address}\n")
    print("-" * 150)
    print(f"{'序号':<4} {'16进制私钥':<66} {'WIF格式(压缩)':<52} {'比特币地址(压缩)'}")
    print("-" * 150)
    
    current_key = start_value
    found = False
    
    for i in range(count):
        # 将整数转换为64字符的16进制字符串
        private_key_hex = format(current_key, '064x')
        
        # 生成比特币地址
        address_compressed = private_key_to_address(private_key_hex, compressed=True)
        
        # 检查是否匹配目标地址
        if address_compressed == target_address:
            # 生成WIF格式
            wif_compressed = private_key_to_wif(private_key_hex, compressed=True)
            
            print(f"{i+1:<4} {private_key_hex} {wif_compressed} {address_compressed}")
            print("\n" + "=" * 150)
            print("找到目标地址!")
            print(f"私钥 (16进制): {private_key_hex}")
            print(f"私钥 (WIF压缩): {wif_compressed}")
            print(f"地址: {address_compressed}")
            found = True
            break
        
        # 每10000个私钥显示一次进度
        if (i + 1) % 10000 == 0:
            print(f"已检查 {i+1} 个私钥...")
        
        current_key += 1
    
    if not found:
        print(f"\n在 {count} 个私钥中未找到目标地址 {target_address}")
        print(f"最后检查的私钥: {hex(current_key-1)}")

# 主程序
if __name__ == "__main__":
    # 2^70 的值
    start_2_70 = 1200000000000000100000
    
    # 要检查的私钥数量
    number_of_keys = 100000  # 可以根据需要调整
    
    # 目标地址
    target_address = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"
    
    print("比特币私钥搜索器 (2^70 到 2^71 范围)")
    print("=" * 150)
    
    find_specific_address(start_2_70, number_of_keys, target_address)
