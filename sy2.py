import os
import json
import secrets
import hashlib
import sys
import time
import signal
from typing import List, Dict, Any

# 依赖检查
try:
    import base58
except ImportError:
    print("错误: 缺少 base58 库，正在安装...")
    os.system(f"{sys.executable} -m pip install base58 -i https://pypi.tuna.tsinghua.edu.cn/simple")
    try:
        import base58
        print("base58 库安装成功!")
    except ImportError:
        print("安装失败，请手动运行: pip install base58")
        sys.exit(1)

try:
    import ecdsa
except ImportError:
    print("错误: 缺少 ecdsa 库，正在安装...")
    os.system(f"{sys.executable} -m pip install ecdsa -i https://pypi.tuna.tsinghua.edu.cn/simple")
    try:
        import ecdsa
        print("ecdsa 库安装成功!")
    except ImportError:
        print("安装失败，请手动运行: pip install ecdsa")
        sys.exit(1)

class BitcoinKeyGenerator:
    def __init__(self, data_file="generated_keys.json"):
        self.data_file = data_file
        self.is_running = True
        self.current_batch = 0
        self.start_time = None
        
        # 设置信号处理，优雅退出
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.generated_data = self._load_data()
        
    def signal_handler(self, signum, frame):
        """处理中断信号，优雅退出"""
        print(f"\n收到中断信号，正在保存数据...")
        self.is_running = False
        self._save_data()
        print("数据已保存，退出程序")
        sys.exit(0)
        
    def _load_data(self) -> dict:
        """加载已生成的数据，增加错误处理"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"已加载历史数据: 总共生成 {data.get('total_generated', 0)} 个密钥")
                    return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"加载数据文件失败: {e}，将创建新文件")
        
        return {
            "total_generated": 0,
            "found_keys": [],
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_range": "2^70 to 2^71"
        }
    
    def _save_data(self):
        """保存数据到文件，增加错误处理"""
        try:
            # 更新最后保存时间
            self.generated_data["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # 使用临时文件避免写入过程中出错
            temp_file = self.data_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.generated_data, f, indent=2, ensure_ascii=False)
            
            # 重命名替换原文件
            if os.path.exists(self.data_file):
                os.remove(self.data_file)
            os.rename(temp_file, self.data_file)
            
        except Exception as e:
            print(f"保存数据失败: {e}")
    
    def private_key_to_wif(self, private_key: int) -> str:
        """将私钥转换为WIF格式，增加错误处理"""
        try:
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
        except Exception as e:
            print(f"WIF转换错误: {e}")
            return None
    
    def private_key_to_address(self, private_key: int) -> str:
        """从私钥生成比特币地址，增加错误处理"""
        try:
            # 使用secp256k1曲线生成公钥
            sk = ecdsa.SigningKey.from_string(private_key.to_bytes(32, 'big'), curve=ecdsa.SECP256k1)
            vk = sk.verifying_key
            
            # 压缩公钥格式
            if vk.pubkey.point.y() % 2 == 0:
                public_key = b'\x02' + vk.pubkey.point.x().to_bytes(32, 'big')
            else:
                public_key = b'\x03' + vk.pubkey.point.x().to_bytes(32, 'big')
            
            # SHA256哈希
            sha256_hash = hashlib.sha256(public_key).digest()
            # RIPEMD160哈希
            ripemd160 = hashlib.new('ripemd160')
            ripemd160.update(sha256_hash)
            ripemd160_hash = ripemd160.digest()
            
            # 添加版本字节（0x00 主网）
            extended_hash = b'\x00' + ripemd160_hash
            # 计算校验和
            checksum_full = hashlib.sha256(hashlib.sha256(extended_hash).digest()).digest()
            checksum = checksum_full[:4]
            # Base58编码
            bitcoin_address = base58.b58encode(extended_hash + checksum).decode('ascii')
            
            return bitcoin_address
        except Exception as e:
            print(f"地址生成错误: {e}")
            return None
    
    def generate_private_key_in_range(self) -> int:
        """在2^70到2^71范围内生成随机私钥"""
        try:
            # 2^70 = 1180591620717411303424
            # 2^71 = 2361183241434822606848
            min_range = 1180591620717411303424
            max_range = 2361183241434822606848
            return secrets.randbelow(max_range - min_range) + min_range
        except Exception as e:
            print(f"私钥生成错误: {e}")
            return None
    
    def generate_and_check_keys(self, batch_size: int = 10000) -> List[Dict[str, Any]]:
        """生成一批密钥并检查是否符合条件"""
        found_keys = []
        successful_generations = 0
        
        for i in range(batch_size):
            if not self.is_running:
                break
                
            try:
                # 生成私钥
                private_key = self.generate_private_key_in_range()
                if private_key is None:
                    continue
                
                # 生成地址
                address = self.private_key_to_address(private_key)
                if address is None:
                    continue
                
                successful_generations += 1
                
                # 检查是否以1PWo3J开头
                if address.startswith('1PWo3J'):
                    wif_key = self.private_key_to_wif(private_key)
                    if wif_key:
                        result = {
                            'private_key_hex': hex(private_key)[2:].zfill(64),
                            'private_key_wif': wif_key,
                            'address': address,
                            'found_time': time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        found_keys.append(result)
                        print(f"🎉 找到符合条件的地址: {address}")
                        
                        # 立即保存重要发现
                        self.generated_data["found_keys"].append(result)
                        self._save_data()
                
                # 更新进度
                if (i + 1) % 1000 == 0:
                    elapsed = time.time() - self.start_time
                    keys_per_sec = (self.current_batch * batch_size + i + 1) / elapsed
                    print(f"批次 {self.current_batch + 1} - 已生成 {i + 1}/{batch_size} 个密钥 "
                          f"(速度: {keys_per_sec:.1f} 密钥/秒)")
            
            except Exception as e:
                print(f"密钥生成过程中出错: {e}")
                continue
        
        return found_keys
    
    def calculate_stats(self):
        """计算并显示统计信息"""
        if not self.generated_data["found_keys"]:
            return "暂无找到的地址"
        
        addresses = [key['address'] for key in self.generated_data["found_keys"]]
        unique_addresses = set(addresses)
        
        return f"找到 {len(self.generated_data['found_keys'])} 个密钥，" \
               f"其中 {len(unique_addresses)} 个唯一地址"
    
    def run_generation(self, batch_size: int = 10000, total_batches: int = None):
        """运行密钥生成过程"""
        self.start_time = time.time()
        self.is_running = True
        
        print(f"开始生成密钥...")
        print(f"目标前缀: 1PWo3J")
        print(f"私钥范围: 2^70 到 2^71")
        print(f"每批次: {batch_size} 个密钥")
        
        if total_batches:
            print(f"总批次: {total_batches}")
        
        batch_count = 0
        
        try:
            while self.is_running and (total_batches is None or batch_count < total_batches):
                self.current_batch = batch_count
                print(f"\n--- 批次 {batch_count + 1} ---")
                
                # 生成并检查密钥
                new_found_keys = self.generate_and_check_keys(batch_size)
                
                # 更新数据
                self.generated_data["total_generated"] += batch_size
                
                # 定期保存进度（每批次）
                self._save_data()
                
                # 显示批次结果
                elapsed = time.time() - self.start_time
                total_keys = self.generated_data["total_generated"]
                keys_per_sec = total_keys / elapsed
                
                print(f"批次完成! 本批找到 {len(new_found_keys)} 个符合条件的地址")
                print(f"累计统计: {self.calculate_stats()}")
                print(f"总生成: {total_keys} 个密钥, 速度: {keys_per_sec:.1f} 密钥/秒")
                
                batch_count += 1
                
                # 如果不是连续运行，显示找到的密钥
                if new_found_keys and total_batches == 1:
                    print("\n本次找到的密钥:")
                    for key_data in new_found_keys:
                        print(f"地址: {key_data['address']}")
                        print(f"WIF私钥: {key_data['private_key_wif']}")
                        print(f"十六进制私钥: {key_data['private_key_hex']}")
                        print("-" * 50)
                
                # 如果指定了总批次，检查是否完成
                if total_batches and batch_count >= total_batches:
                    break
                    
        except Exception as e:
            print(f"生成过程中发生错误: {e}")
        finally:
            # 最终保存
            self._save_data()
            total_elapsed = time.time() - self.start_time
            print(f"\n任务完成! 总运行时间: {total_elapsed:.1f} 秒")
            print(f"最终统计: {self.calculate_stats()}")
    
    def show_statistics(self):
        """显示详细统计信息"""
        print("\n" + "="*50)
        print("📊 统计信息")
        print("="*50)
        print(f"总共生成的密钥数量: {self.generated_data['total_generated']:,}")
        print(f"找到的符合条件的地址数量: {len(self.generated_data['found_keys'])}")
        print(f"数据最后更新: {self.generated_data.get('last_update', '未知')}")
        
        if self.generated_data['found_keys']:
            print(f"\n找到的地址列表:")
            for i, key_data in enumerate(self.generated_data['found_keys'], 1):
                print(f"{i}. 地址: {key_data['address']}")
                print(f"   WIF私钥: {key_data['private_key_wif']}")
                print(f"   发现时间: {key_data.get('found_time', '未知')}")
                print("-" * 40)

def main():
    print("比特币密钥生成器 - 腾讯云优化版")
    print("目标: 寻找以 '1PWo3J' 开头的比特币地址")
    
    generator = BitcoinKeyGenerator()
    
    while True:
        print("\n" + "="*50)
        print("🔑 比特币密钥生成器")
        print("="*50)
        print("1. 单次生成10000个密钥")
        print("2. 连续生成密钥（直到手动停止）")
        print("3. 自定义单次生成数量")
        print("4. 自定义连续生成批次")
        print("5. 显示统计信息")
        print("6. 退出程序")
        print("\n提示: 使用 Ctrl+C 可以安全中断生成过程")
        
        try:
            choice = input("\n请选择操作 (1-6): ").strip()
            
            if choice == '1':
                generator.run_generation(10000, 1)
            elif choice == '2':
                print("开始连续生成，使用 Ctrl+C 停止...")
                generator.run_generation(10000)
            elif choice == '3':
                try:
                    count = int(input("请输入要生成的密钥数量: "))
                    if 1000 <= count <= 1000000:
                        generator.run_generation(count, 1)
                    else:
                        print("数量范围应在 1000 到 1000000 之间")
                except ValueError:
                    print("请输入有效的数字！")
            elif choice == '4':
                try:
                    batches = int(input("请输入要生成的批次数量: "))
                    if 1 <= batches <= 1000:
                        generator.run_generation(10000, batches)
                    else:
                        print("批次范围应在 1 到 1000 之间")
                except ValueError:
                    print("请输入有效的数字！")
            elif choice == '5':
                generator.show_statistics()
            elif choice == '6':
                print("再见！")
                break
            else:
                print("无效的选择，请重新输入！")
        except KeyboardInterrupt:
            print("\n用户中断操作")
            continue

if __name__ == "__main__":
    main()
