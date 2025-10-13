import os
import json
import secrets
import hashlib
import base58
import ecdsa
from typing import Tuple, Optional

class BitcoinKeyGenerator:
    def __init__(self, data_file="generated_keys.json"):
        self.data_file = data_file
        self.generated_data = self._load_data()
        
    def _load_data(self) -> dict:
        """加载已生成的数据"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return {
            "total_generated": 0,
            "found_keys": [],
            "last_private_key": None
        }
    
    def _save_data(self):
        """保存数据到文件"""
        with open(self.data_file, 'w') as f:
            json.dump(self.generated_data, f, indent=2)
    
    def private_key_to_wif(self, private_key: int) -> str:
        """将私钥转换为WIF格式"""
        # 添加前缀0x80（主网）和压缩标志0x01
        extended_key = b'\x80' + private_key.to_bytes(32, 'big') + b'\x01'
        # 双重SHA256哈希
        first_hash = hashlib.sha256(extended_key).digest()
        second_hash = hashlib.sha256(first_hash).digest()
        # 添加校验和
        checksum = second_hash[:4]
        final_key = extended_key + checksum
        # Base58编码
        return base58.b58encode(final_key).decode('ascii')
    
    def private_key_to_address(self, private_key: int) -> str:
        """从私钥生成比特币地址"""
        # 使用secp256k1曲线生成公钥
        sk = ecdsa.SigningKey.from_string(private_key.to_bytes(32, 'big'), curve=ecdsa.SECP256k1)
        vk = sk.verifying_key
        # 压缩公钥格式
        public_key = b'\x02' + vk.pubkey.point.x().to_bytes(32, 'big') if vk.pubkey.point.y() % 2 == 0 else b'\x03' + vk.pubkey.point.x().to_bytes(32, 'big')
        
        # SHA256哈希
        sha256_hash = hashlib.sha256(public_key).digest()
        # RIPEMD160哈希
        ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
        # 添加版本字节（0x00 主网）
        extended_hash = b'\x00' + ripemd160_hash
        # 计算校验和
        checksum = hashlib.sha256(hashlib.sha256(extended_hash).digest()).digest()[:4]
        # Base58编码
        bitcoin_address = base58.b58encode(extended_hash + checksum).decode('ascii')
        
        return bitcoin_address
    
    def generate_private_key_in_range(self) -> int:
        """在2^70到2^71范围内生成随机私钥"""
        min_range = 1680591620717411303424
        max_range = 2361183241434822606848
        return secrets.randbelow(max_range - min_range + 1) + min_range
    
    def generate_and_check_keys(self, batch_size: int = 10000) -> list:
        """生成一批密钥并检查是否符合条件"""
        found_keys = []
        
        for i in range(batch_size):
            # 生成私钥
            private_key = self.generate_private_key_in_range()
            
            # 生成地址
            address = self.private_key_to_address(private_key)
            
            # 检查是否以1PWo开头
            if address.startswith('1PWo3J'):
                wif_key = self.private_key_to_wif(private_key)
                result = {
                    'private_key_hex': hex(private_key)[2:].zfill(64),
                    'private_key_wif': wif_key,
                    'address': address
                }
                found_keys.append(result)
                print(f"找到符合条件的地址: {address}")
            
            # 更新进度
            if (i + 1) % 10000 == 0:
                print(f"已生成 {i + 1}/{batch_size} 个密钥")
        
        return found_keys
    
    def run_generation(self, batch_size: int = 10000):
        """运行密钥生成过程"""
        print(f"开始生成 {batch_size} 个密钥...")
        
        # 生成并检查密钥
        new_found_keys = self.generate_and_check_keys(batch_size)
        
        # 更新数据
        self.generated_data["total_generated"] += batch_size
        self.generated_data["found_keys"].extend(new_found_keys)
        
        # 保存数据
        self._save_data()
        
        # 输出结果
        print(f"\n完成！总共已生成: {self.generated_data['total_generated']} 个密钥")
        print(f"本次找到 {len(new_found_keys)} 个符合条件的地址")
        print(f"累计找到 {len(self.generated_data['found_keys'])} 个符合条件的地址")
        
        # 显示找到的密钥
        if new_found_keys:
            print("\n本次找到的密钥:")
            for key_data in new_found_keys:
                print(f"地址: {key_data['address']}")
                print(f"WIF私钥: {key_data['private_key_wif']}")
                print(f"十六进制私钥: {key_data['private_key_hex']}")
                print("-" * 50)
    
    def show_statistics(self):
        """显示统计信息"""
        print("\n=== 统计信息 ===")
        print(f"总共生成的密钥数量: {self.generated_data['total_generated']}")
        print(f"找到的符合条件的地址数量: {len(self.generated_data['found_keys'])}")
        
        if self.generated_data['found_keys']:
            print("\n找到的地址列表:")
            for key_data in self.generated_data['found_keys']:
                print(f"地址: {key_data['address']}")
                print(f"WIF私钥: {key_data['private_key_wif']}")
                print("-" * 30)

def main():
    generator = BitcoinKeyGenerator()
    
    while True:
        print("\n=== 比特币密钥生成器 ===")
        print("1. 生成10000个新密钥")
        print("2. 生成自定义数量的密钥")
        print("3. 显示统计信息")
        print("4. 退出")
        
        choice = input("请选择操作 (1-4): ").strip()
        
        if choice == '1':
            generator.run_generation(10000)
        elif choice == '2':
            try:
                count = int(input("请输入要生成的密钥数量: "))
                if count > 0:
                    generator.run_generation(count)
                else:
                    print("请输入正数！")
            except ValueError:
                print("请输入有效的数字！")
        elif choice == '3':
            generator.show_statistics()
        elif choice == '4':
            print("再见！")
            break
        else:
            print("无效的选择，请重新输入！")

if __name__ == "__main__":
    main()
