import os
import json
import secrets
import hashlib
import sys
import time
import signal
import multiprocessing as mp
import threading
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# 依赖检查
def install_dependencies():
    """安装必要的依赖包"""
    dependencies = [
        "base58",
        "ecdsa",
    ]
    
    # 检查 GPU 加速库
    gpu_dependencies = []
    try:
        import cupy
        print("CuPy 已安装")
    except ImportError:
        gpu_dependencies.append("cupy-cuda11x")
        print("将安装 CuPy")
    
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import pycuda.compiler
        print("PyCUDA 已安装")
    except ImportError:
        gpu_dependencies.append("pycuda")
        print("将安装 PyCUDA")
    
    all_deps = dependencies + gpu_dependencies
    
    for dep in all_deps:
        try:
            if dep == "base58":
                import base58
            elif dep == "ecdsa":
                import ecdsa
            elif dep.startswith("cupy"):
                import cupy
            elif dep == "pycuda":
                import pycuda.autoinit
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
                elif dep == "pycuda":
                    import pycuda.autoinit
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
    HAS_CUPY = True
    print("CuPy GPU 加速已启用")
except ImportError:
    HAS_CUPY = False
    print("CuPy 不可用")

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
    HAS_PYCUDA = True
    print("PyCUDA GPU 加速已启用")
except ImportError:
    HAS_PYCUDA = False
    print("PyCUDA 不可用")

# PyCUDA 内核代码
CUDA_KERNEL = """
#include <stdint.h>

// SHA256 常量
__constant__ uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA256 函数
__device__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// SHA256 计算内核
__global__ void sha256_kernel(uint8_t *input, uint32_t input_len, uint8_t *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程处理一个输入
    uint8_t *in = input + idx * input_len;
    uint8_t *out = output + idx * 32;
    
    // SHA256 实现...
    // 这里简化实现，实际使用时需要完整实现
}

// 随机数生成内核
__global__ void generate_random_keys(uint64_t *keys, uint64_t min_val, uint64_t max_val, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // 简单的随机数生成（实际使用时需要更好的随机性）
        keys[idx] = min_val + (idx % (max_val - min_val));
    }
}
"""

class GPUBitcoinKeyGenerator:
    def __init__(self, data_file="gpu_keys.json", use_gpu=True):
        self.data_file = data_file
        self.use_gpu = use_gpu and (HAS_CUPY or HAS_PYCUDA)
        self.is_running = True
        self.current_batch = 0
        self.start_time = None
        self.total_found = 0
        self.lock = threading.Lock()
        
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
            if HAS_PYCUDA:
                # 获取GPU信息
                device = cuda.Device(0)
                context = device.make_context()
                attrs = device.get_attributes()
                
                print(f"PyCUDA GPU: {device.name()}")
                print(f"  计算能力: {device.compute_capability()}")
                print(f"  全局内存: {device.total_memory() / 1024**3:.1f} GB")
                print(f"  多处理器: {attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT]}")
                
                # 编译CUDA内核
                try:
                    self.cuda_module = SourceModule(CUDA_KERNEL)
                    self.sha256_kernel = self.cuda_module.get_function("sha256_kernel")
                    self.random_keys_kernel = self.cuda_module.get_function("generate_random_keys")
                    print("CUDA内核编译成功")
                except Exception as e:
                    print(f"CUDA内核编译失败: {e}")
                    self.use_gpu = False
            
            if HAS_CUPY:
                # 获取CuPy GPU信息
                gpu_count = cp.cuda.runtime.getDeviceCount()
                print(f"CuPy GPU设备: {gpu_count}")
                
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
            
            if self.use_gpu and HAS_CUPY:
                # 使用CuPy进行GPU加速的SHA256计算
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
        """从私钥生成比特币地址"""
        try:
            # 使用secp256k1曲线生成公钥
            sk = ecdsa.SigningKey.from_string(private_key.to_bytes(32, 'big'), curve=ecdsa.SECP256k1)
            vk = sk.verifying_key
            
            # 压缩公钥格式
            x = vk.pubkey.point.x()
            y = vk.pubkey.point.y()
            
            if y % 2 == 0:
                public_key = b'\x02' + x.to_bytes(32, 'big')
            else:
                public_key = b'\x03' + x.to_bytes(32, 'big')
            
            # 计算哈希
            if self.use_gpu and HAS_CUPY:
                # 使用CuPy进行GPU加速的SHA256计算
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
            if self.use_gpu and HAS_CUPY:
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
    
    def generate_private_keys_gpu(self, batch_size: int) -> List[int]:
        """使用GPU生成私钥"""
        min_range = 1180591620717411303424  # 2^70
        max_range = 2361183241434822606848  # 2^71
        
        if self.use_gpu:
            try:
                if HAS_CUPY:
                    # 使用CuPy生成随机数
                    random_keys_gpu = cp.random.randint(min_range, max_range, batch_size, dtype=cp.int64)
                    private_keys = cp.asnumpy(random_keys_gpu).tolist()
                    return private_keys
                
                elif HAS_PYCUDA:
                    # 使用PyCUDA生成随机数
                    keys_gpu = gpuarray.zeros(batch_size, dtype=cp.uint64)
                    block_size = 256
                    grid_size = (batch_size + block_size - 1) // block_size
                    
                    self.random_keys_kernel(
                        keys_gpu, 
                        cp.uint64(min_range), 
                        cp.uint64(max_range), 
                        cp.int32(batch_size),
                        block=(block_size, 1, 1), 
                        grid=(grid_size, 1)
                    )
                    
                    private_keys = keys_gpu.get().tolist()
                    return private_keys
                    
            except Exception as e:
                print(f"GPU随机数生成失败: {e}")
        
        # 回退到CPU生成
        return [secrets.randbelow(max_range - min_range) + min_range for _ in range(batch_size)]
    
    def process_private_key(self, private_key: int) -> Dict[str, Any]:
        """处理单个私钥"""
        try:
            address = self.private_key_to_address(private_key)
            if address and address.startswith('1PWo3J'):
                wif_key = self.private_key_to_wif(private_key)
                if wif_key:
                    return {
                        'private_key_hex': hex(private_key)[2:].zfill(64),
                        'private_key_wif': wif_key,
                        'address': address,
                        'found_time': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
        except Exception as e:
            pass
        return None
    
    def process_batch_gpu_parallel(self, batch_size: int) -> List[Dict[str, Any]]:
        """使用GPU和多线程处理批次"""
        found_keys = []
        
        # 使用GPU生成私钥
        private_keys = self.generate_private_keys_gpu(batch_size)
        
        # 使用多线程处理私钥
        max_workers = min(mp.cpu_count(), 16)  # 最多16个线程
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_key = {executor.submit(self.process_private_key, key): key for key in private_keys}
            
            # 收集结果
            for future in as_completed(future_to_key):
                if not self.is_running:
                    break
                    
                try:
                    result = future.result(timeout=30)  # 30秒超时
                    if result:
                        with self.lock:
                            found_keys.append(result)
                            self.total_found += 1
                            print(f"🎉 找到地址 #{self.total_found}: {result['address']}")
                            
                            # 立即保存重要发现
                            self.generated_data["found_keys"].append(result)
                            self._save_data()
                except Exception as e:
                    continue
        
        return found_keys
    
    def run_generation(self, batch_size: int = 10000, total_batches: int = None):
        """运行密钥生成过程"""
        self.start_time = time.time()
        self.is_running = True
        
        print(f"\n🚀 开始GPU加速密钥生成")
        print(f"目标前缀: 1PWo3J")
        print(f"私钥范围: 2^70 到 2^71")
        print(f"批次大小: {batch_size}")
        print(f"GPU加速: {'是' if self.use_gpu else '否'}")
        
        if HAS_CUPY:
            print(f"CuPy: 已启用")
        if HAS_PYCUDA:
            print(f"PyCUDA: 已启用")
        
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
                found_keys = self.process_batch_gpu_parallel(batch_size)
                
                # 更新数据
                self.generated_data["total_generated"] += batch_size
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
                
                # 定期保存进度
                if (batch_count + 1) % 5 == 0:
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
        print(f"CuPy: {'已启用' if HAS_CUPY else '未启用'}")
        print(f"PyCUDA: {'已启用' if HAS_PYCUDA else '未启用'}")
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
    use_gpu = HAS_CUPY or HAS_PYCUDA
    if not use_gpu:
        print("⚠️  未检测到GPU支持，将使用CPU模式")
    
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
        print(f"\n当前模式: {'GPU加速' if generator.use_gpu else 'CPU'}")
        print(f"CuPy: {'已启用' if HAS_CUPY else '未启用'}")
        print(f"PyCUDA: {'已启用' if HAS_PYCUDA else '未启用'}")
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
    main()
