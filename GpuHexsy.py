import os
import json
import secrets
import hashlib
import sys
import time
import signal
import multiprocessing as mp
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

# 依赖检查
def install_dependencies():
    """安装必要的依赖包"""
    dependencies = [
        "base58",
        "ecdsa",
    ]
    
    # 检查是否可能使用GPU加速的库
    gpu_dependencies = []
    try:
        import cupy
        print("CuPy 已安装，将使用 GPU 加速")
    except ImportError:
        gpu_dependencies.append("cupy-cuda11x")
        print("将安装 CuPy 以启用 GPU 加速")
    
    all_deps = dependencies + gpu_dependencies
    
    for dep in all_deps:
        try:
            if dep == "base58":
                import base58
            elif dep == "ecdsa":
                import ecdsa
            elif dep.startswith("cupy"):
                import cupy
            print(f"✓ {dep} 已安装")
        except ImportError:
            print(f"正在安装 {dep}...")
            os.system(f"{sys.executable} -m pip install {dep} -i https://pypi.tuna.tsinghua.edu.cn/simple")
            try:
                if dep == "base58":
                    import base58
                elif dep == "ecdsa":
                    import ecdsa
                elif dep.startswith("cupy"):
                    import cupy
                print(f"✓ {dep} 安装成功")
            except ImportError:
                print(f"✗ {dep} 安装失败")

# 安装依赖
install_dependencies()

# 导入主库
import base58
import ecdsa

# 尝试导入GPU加速库
try:
    import cupy as cp
    HAS_GPU = True
    print("GPU 加速已启用")
except ImportError:
    HAS_GPU = False
    print("GPU 加速不可用，将使用 CPU")

class GPUBitcoinKeyGenerator:
    def __init__(self, data_file="gpu_generated_keys.json", use_gpu=True):
        self.data_file = data_file
        self.use_gpu = use_gpu and HAS_GPU
        self.is_running = True
        self.current_batch = 0
        self.start_time = None
        self.total_found = 0
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.generated_data = self._load_data()
        
        # GPU初始化
        if self.use_gpu:
            self._init_gpu()
    
    def _init_gpu(self):
        """初始化GPU环境"""
        try:
            # 获取GPU信息
            gpu_count = cp.cuda.runtime.getDeviceCount()
            print(f"检测到 {gpu_count} 个GPU设备")
            
            for i in range(gpu_count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                print(f"GPU {i}: {props['name'].decode()}")
                print(f"  计算能力: {props['major']}.{props['minor']}")
                print(f"  全局内存: {props['totalGlobalMem'] / 1024**3:.1f} GB")
            
            # 设置当前GPU
            cp.cuda.Device(0).use()
            
        except Exception as e:
            print(f"GPU初始化失败: {e}")
            self.use_gpu = False
    
    def signal_handler(self, signum, frame):
        """处理中断信号"""
        print(f"\n收到中断信号，正在保存数据...")
        self.is_running = False
        self._save_data()
        print("数据已保存，退出程序")
        sys.exit(0)
        
    def _load_data(self) -> dict:
        """加载已生成的数据"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"已加载历史数据: 总共生成 {data.get('total_generated', 0)} 个密钥")
                    return data
        except Exception as e:
            print(f"加载数据文件失败: {e}")
        
        return {
            "total_generated": 0,
            "found_keys": [],
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_range": "2^70 to 2^71",
            "gpu_accelerated": self.use_gpu
        }
    
    def _save_data(self):
        """保存数据到文件"""
        try:
            self.generated_data["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.generated_data["gpu_accelerated"] = self.use_gpu
            
            temp_file = self.data_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.generated_data, f, indent=2, ensure_ascii=False)
            
            if os.path.exists(self.data_file):
                os.remove(self.data_file)
            os.rename(temp_file, self.data_file)
            
        except Exception as e:
            print(f"保存数据失败: {e}")
    
    def private_key_to_wif(self, private_key: int) -> str:
        """将私钥转换为WIF格式"""
        try:
            extended_key = b'\x80' + private_key.to_bytes(32, 'big') + b'\x01'
            
            if self.use_gpu:
                # 使用GPU进行SHA256计算
                extended_key_gpu = cp.asarray(bytearray(extended_key))
                first_hash = cp.asnumpy(cp.sha256(extended_key_gpu))
                second_hash = cp.asnumpy(cp.sha256(cp.array(first_hash)))
                checksum = second_hash[:4]
            else:
                # CPU计算
                first_hash = hashlib.sha256(extended_key).digest()
                second_hash = hashlib.sha256(first_hash).digest()
                checksum = second_hash[:4]
            
            final_key = extended_key + checksum
            return base58.b58encode(final_key).decode('ascii')
        except Exception as e:
            print(f"WIF转换错误: {e}")
            return None
    
    def private_key_to_address(self, private_key: int) -> str:
        """从私钥生成比特币地址（修复压缩公钥错误）"""
        try:
            # 使用secp256k1曲线生成公钥
            sk = ecdsa.SigningKey.from_string(private_key.to_bytes(32, 'big'), curve=ecdsa.SECP256k1)
            vk = sk.verifying_key
            
            # 修复压缩公钥格式 - 确保使用正确的点坐标
            x = vk.pubkey.point.x()
            y = vk.pubkey.point.y()
            
            # 压缩公钥格式：0x02 如果 y 是偶数，0x03 如果 y 是奇数
            if y % 2 == 0:
                public_key = b'\x02' + x.to_bytes(32, 'big')
            else:
                public_key = b'\x03' + x.to_bytes(32, 'big')
            
            # 计算哈希
            if self.use_gpu:
                # GPU加速的SHA256计算
                public_key_gpu = cp.asarray(bytearray(public_key))
                sha256_hash = cp.asnumpy(cp.sha256(public_key_gpu))
            else:
                # CPU计算
                sha256_hash = hashlib.sha256(public_key).digest()
            
            # RIPEMD160仍然需要在CPU上计算
            ripemd160 = hashlib.new('ripemd160')
            ripemd160.update(sha256_hash)
            ripemd160_hash = ripemd160.digest()
            
            # 添加版本字节（0x00 主网）
            extended_hash = b'\x00' + ripemd160_hash
            
            # 计算校验和
            if self.use_gpu:
                extended_hash_gpu = cp.asarray(bytearray(extended_hash))
                checksum_full = cp.asnumpy(cp.sha256(cp.sha256(extended_hash_gpu)))
                checksum = checksum_full[:4]
            else:
                checksum_full = hashlib.sha256(hashlib.sha256(extended_hash).digest()).digest()
                checksum = checksum_full[:4]
            
            # Base58编码
            bitcoin_address = base58.b58encode(extended_hash + checksum).decode('ascii')
            return bitcoin_address
            
        except Exception as e:
            print(f"地址生成错误: {e}")
            return None
    
    def private_key_to_address_batch(self, private_keys: List[int]) -> List[str]:
        """批量生成地址 - 修复版本"""
        addresses = []
        
        for private_key in private_keys:
            address = self.private_key_to_address(private_key)
            addresses.append(address)
        
        return addresses
    
    def generate_private_keys_batch(self, batch_size: int) -> List[int]:
        """批量生成私钥"""
        min_range = 1180591620717411303424  # 2^70
        max_range = 2361183241434822606848  # 2^71
        
        if self.use_gpu:
            try:
                # 使用GPU生成随机数
                random_keys_gpu = cp.random.randint(min_range, max_range, batch_size, dtype=cp.int64)
                private_keys = cp.asnumpy(random_keys_gpu).tolist()
            except Exception as e:
                print(f"GPU随机数生成失败，回退到CPU: {e}")
                private_keys = [secrets.randbelow(max_range - min_range) + min_range 
                               for _ in range(batch_size)]
        else:
            # CPU生成随机数
            private_keys = [secrets.randbelow(max_range - min_range) + min_range 
                           for _ in range(batch_size)]
        
        return private_keys
    
    def process_chunk(self, chunk_data):
        """处理数据块 - 用于多进程"""
        chunk_private_keys, chunk_id, target_prefix = chunk_data
        found_keys = []
        
        for private_key in chunk_private_keys:
            try:
                address = self.private_key_to_address(private_key)
                if address and address.startswith(target_prefix):
                    wif_key = self.private_key_to_wif(private_key)
                    if wif_key:
                        result = {
                            'private_key_hex': hex(private_key)[2:].zfill(64),
                            'private_key_wif': wif_key,
                            'address': address,
                            'found_time': time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        found_keys.append(result)
            except Exception as e:
                continue
        
        return found_keys
    
    def process_batch_parallel(self, batch_size: int, target_prefix: str = "1PWo3J") -> List[Dict[str, Any]]:
        """并行处理批次 - 使用多进程提高CPU利用率"""
        found_keys = []
        
        # 生成私钥批次
        private_keys = self.generate_private_keys_batch(batch_size)
        
        # 根据CPU核心数确定进程数
        cpu_count = min(mp.cpu_count(), 8)  # 最多使用8个进程
        chunk_size = max(batch_size // cpu_count, 1000)
        
        # 分割数据为块
        chunks = []
        for i in range(0, len(private_keys), chunk_size):
            chunk = private_keys[i:i + chunk_size]
            chunks.append((chunk, i // chunk_size, target_prefix))
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = [executor.submit(self.process_chunk, chunk) for chunk in chunks]
            
            for future in as_completed(futures):
                try:
                    chunk_found_keys = future.result(timeout=300)  # 5分钟超时
                    found_keys.extend(chunk_found_keys)
                except Exception as e:
                    print(f"处理块时出错: {e}")
                    continue
        
        return found_keys
    
    def run_generation(self, batch_size: int = 10000, total_batches: int = None):
        """运行密钥生成过程"""
        self.start_time = time.time()
        self.is_running = True
        
        print(f"\n🚀 开始{'GPU加速' if self.use_gpu else 'CPU多进程'}密钥生成")
        print(f"目标前缀: 1PWo3J")
        print(f"私钥范围: 2^70 到 2^71")
        print(f"批次大小: {batch_size}")
        print(f"GPU加速: {'是' if self.use_gpu else '否'}")
        
        if total_batches:
            print(f"总批次: {total_batches}")
        
        batch_count = 0
        total_keys_generated = 0
        
        try:
            while self.is_running and (total_batches is None or batch_count < total_batches):
                self.current_batch = batch_count
                batch_start_time = time.time()
                
                print(f"\n--- 批次 {batch_count + 1} ---")
                
                # 处理批次
                found_keys = self.process_batch_parallel(batch_size)
                
                # 更新数据
                self.generated_data["total_generated"] += batch_size
                self.generated_data["found_keys"].extend(found_keys)
                total_keys_generated += batch_size
                
                # 显示批次结果
                batch_time = time.time() - batch_start_time
                keys_per_sec = batch_size / batch_time
                
                total_elapsed = time.time() - self.start_time
                overall_speed = total_keys_generated / total_elapsed
                
                print(f"批次完成! 耗时: {batch_time:.1f}秒, 速度: {keys_per_sec:.1f} 密钥/秒")
                print(f"本批找到: {len(found_keys)} 个符合条件的地址")
                print(f"累计找到: {len(self.generated_data['found_keys'])} 个地址")
                print(f"总速度: {overall_speed:.1f} 密钥/秒")
                
                # 显示找到的密钥
                for key_data in found_keys:
                    self.total_found += 1
                    print(f"🎉 找到地址 #{self.total_found}: {key_data['address']}")
                
                # 定期保存进度
                if (batch_count + 1) % 5 == 0 or found_keys:
                    self._save_data()
                    print("进度已保存")
                
                batch_count += 1
                
                # 如果指定了总批次，检查是否完成
                if total_batches and batch_count >= total_batches:
                    break
                    
        except Exception as e:
            print(f"生成过程中发生错误: {e}")
        except KeyboardInterrupt:
            print("\n用户中断生成过程")
        finally:
            # 最终保存
            self._save_data()
            total_elapsed = time.time() - self.start_time
            print(f"\n任务完成! 总运行时间: {total_elapsed:.1f} 秒")
            print(f"总生成密钥: {total_keys_generated:,}")
            print(f"平均速度: {overall_speed:.1f} 密钥/秒")
            print(f"找到地址总数: {len(self.generated_data['found_keys'])}")
    
    def show_statistics(self):
        """显示统计信息"""
        print("\n" + "="*60)
        print("📊 GPU比特币密钥生成器 - 统计信息")
        print("="*60)
        print(f"总共生成的密钥数量: {self.generated_data['total_generated']:,}")
        print(f"找到的符合条件的地址数量: {len(self.generated_data['found_keys'])}")
        print(f"GPU加速: {'是' if self.generated_data.get('gpu_accelerated', False) else '否'}")
        print(f"数据最后更新: {self.generated_data.get('last_update', '未知')}")
        
        if self.generated_data['found_keys']:
            print(f"\n最近找到的地址:")
            for i, key_data in enumerate(self.generated_data['found_keys'][-5:], 1):
                print(f"{i}. 地址: {key_data['address']}")
                print(f"   WIF私钥: {key_data['private_key_wif']}")
                print(f"   发现时间: {key_data.get('found_time', '未知')}")
                print("-" * 50)

def main():
    print("🚀 GPU加速比特币密钥生成器 - 腾讯云优化版")
    print("目标: 寻找以 '1PWo3J' 开头的比特币地址")
    
    # 检测GPU可用性
    use_gpu = HAS_GPU
    if not HAS_GPU:
        print("⚠️  未检测到GPU支持，将使用CPU多进程模式")
    
    generator = GPUBitcoinKeyGenerator(use_gpu=use_gpu)
    
    while True:
        print("\n" + "="*50)
        print("🔑 GPU比特币密钥生成器")
        print("="*50)
        print("1. 快速生成 (1万密钥/批次)")
        print("2. 高性能生成 (10万密钥/批次)") 
        print("3. 自定义生成参数")
        print("4. 显示统计信息")
        print("5. 退出程序")
        print(f"\n当前模式: {'GPU加速' if generator.use_gpu else 'CPU多进程'}")
        print("提示: 使用 Ctrl+C 安全中断")
        
        try:
            choice = input("\n请选择操作 (1-5): ").strip()
            
            if choice == '1':
                generator.run_generation(batch_size=10000, total_batches=10)
            elif choice == '2':
                generator.run_generation(batch_size=100000, total_batches=10)
            elif choice == '3':
                try:
                    batch_size = int(input("请输入每批次密钥数量: "))
                    total_batches = int(input("请输入总批次数量: "))
                    if batch_size > 0 and total_batches > 0:
                        generator.run_generation(batch_size=batch_size, total_batches=total_batches)
                    else:
                        print("请输入正数！")
                except ValueError:
                    print("请输入有效的数字！")
            elif choice == '4':
                generator.show_statistics()
            elif choice == '5':
                print("再见！")
                break
            else:
                print("无效的选择，请重新输入！")
        except KeyboardInterrupt:
            print("\n用户中断操作")
            continue

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()
