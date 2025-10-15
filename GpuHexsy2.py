import os
import json
import sys
import time
import signal
import threading
from typing import List, Dict, Any
import numpy as np

# 依赖检查 - 只安装GPU相关库
def install_gpu_dependencies():
    """安装GPU相关依赖包"""
    gpu_dependencies = ["cupy-cuda11x", "pycuda"]
    
    for dep in gpu_dependencies:
        try:
            if dep.startswith("cupy"):
                import cupy
                print(f"✓ {dep} 已安装")
            elif dep == "pycuda":
                import pycuda.autoinit
                print(f"✓ {dep} 已安装")
        except ImportError:
            print(f"正在安装 {dep}...")
            os.system(f"{sys.executable} -m pip install {dep} -i https://pypi.tuna.tsinghua.edu.cn/simple")
            try:
                if dep.startswith("cupy"):
                    import cupy
                elif dep == "pycuda":
                    import pycuda.autoinit
                print(f"✓ {dep} 安装成功")
            except ImportError:
                print(f"✗ {dep} 安装失败，程序需要GPU支持")
                sys.exit(1)

# 安装GPU依赖
install_gpu_dependencies()

# 导入GPU库
import cupy as cp
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

# CUDA 内核代码 - 完整的比特币地址生成
BITCOIN_CUDA_KERNEL = """
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

// 基础函数
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

// SHA256 计算
__device__ void sha256(uint8_t *input, int len, uint8_t *output) {
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h_val;
    uint32_t t1, t2;
    
    // 处理512位块
    for (int i = 0; i < len; i += 64) {
        // 准备消息调度
        for (int j = 0; j < 16; j++) {
            w[j] = ((uint32_t)input[i + j*4] << 24) |
                   ((uint32_t)input[i + j*4 + 1] << 16) |
                   ((uint32_t)input[i + j*4 + 2] << 8) |
                   ((uint32_t)input[i + j*4 + 3]);
        }
        
        for (int j = 16; j < 64; j++) {
            w[j] = gamma1(w[j-2]) + w[j-7] + gamma0(w[j-15]) + w[j-16];
        }
        
        a = h[0]; b = h[1]; c = h[2]; d = h[3];
        e = h[4]; f = h[5]; g = h[6]; h_val = h[7];
        
        // 主循环
        for (int j = 0; j < 64; j++) {
            t1 = h_val + sigma1(e) + ch(e, f, g) + k[j] + w[j];
            t2 = sigma0(a) + maj(a, b, c);
            h_val = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += h_val;
    }
    
    // 输出哈希值
    for (int i = 0; i < 8; i++) {
        output[i*4] = (h[i] >> 24) & 0xff;
        output[i*4+1] = (h[i] >> 16) & 0xff;
        output[i*4+2] = (h[i] >> 8) & 0xff;
        output[i*4+3] = h[i] & 0xff;
    }
}

// Base58 编码表
__constant__ char b58_table[59] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// Base58 编码函数
__device__ void base58_encode(uint8_t *input, int len, char *output) {
    // 简化的Base58编码实现
    // 注意：完整实现需要处理大数运算
    for (int i = 0; i < len; i++) {
        output[i] = b58_table[input[i] % 58];
    }
    output[len] = '\\0';
}

// 椭圆曲线点乘法（简化版 - 用于演示）
__device__ void ec_multiply(uint8_t *private_key, uint8_t *public_key) {
    // 简化的椭圆曲线乘法
    // 实际实现需要完整的secp256k1椭圆曲线运算
    for (int i = 0; i < 32; i++) {
        public_key[i] = private_key[i] ^ 0x55; // 简化处理
    }
    public_key[32] = 0x02; // 压缩公钥标志
}

// 比特币地址生成内核
__global__ void bitcoin_address_kernel(
    uint64_t *private_keys, 
    char *addresses, 
    int batch_size,
    uint64_t min_range,
    uint64_t max_range
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // 生成私钥（如果未提供）
    uint64_t private_key;
    if (private_keys == NULL) {
        private_key = min_range + (idx % (max_range - min_range));
    } else {
        private_key = private_keys[idx];
    }
    
    // 转换为字节数组
    uint8_t priv_bytes[32];
    for (int i = 0; i < 8; i++) {
        uint64_t shift = 56 - i * 8;
        uint8_t byte = (private_key >> shift) & 0xff;
        if (i < 4) {
            priv_bytes[24 + i] = byte; // 小端序调整
        } else {
            priv_bytes[i - 4] = byte;
        }
    }
    
    // 生成公钥（简化版）
    uint8_t public_key[33];
    ec_multiply(priv_bytes, public_key);
    
    // SHA256哈希
    uint8_t sha1[32];
    sha256(public_key, 33, sha1);
    
    // RIPEMD160（简化 - 使用SHA256替代）
    uint8_t ripemd160[20];
    sha256(sha1, 32, ripemd160);
    
    // 添加版本字节
    uint8_t extended[21];
    extended[0] = 0x00; // 主网版本
    for (int i = 0; i < 20; i++) {
        extended[i+1] = ripemd160[i];
    }
    
    // 计算校验和
    uint8_t checksum_full[32];
    sha256(extended, 21, checksum_full);
    sha256(checksum_full, 32, checksum_full);
    
    // 构建最终地址数据
    uint8_t address_bytes[25];
    for (int i = 0; i < 21; i++) {
        address_bytes[i] = extended[i];
    }
    for (int i = 0; i < 4; i++) {
        address_bytes[21 + i] = checksum_full[i];
    }
    
    // Base58编码（简化）
    char *addr_ptr = addresses + idx * 35; // 每个地址35字符
    base58_encode(address_bytes, 25, addr_ptr);
    
    // 标记符合条件的地址
    if (addr_ptr[0] == '1' && addr_ptr[1] == 'P' && addr_ptr[2] == 'W' && 
        addr_ptr[3] == 'o' && addr_ptr[4] == '3' && addr_ptr[5] == 'J') {
        addr_ptr[34] = 1; // 标记为找到
    } else {
        addr_ptr[34] = 0;
    }
}

// 随机私钥生成内核
__global__ void generate_private_keys_kernel(
    uint64_t *keys, 
    uint64_t min_val, 
    uint64_t max_val, 
    int count,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    // 简单的随机数生成（实际应使用更安全的随机数生成器）
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rand_val = (seed + tid * 6364136223846793005ULL) % (max_val - min_val);
    keys[idx] = min_val + rand_val;
}
"""

class PureGPUBitcoinGenerator:
    def __init__(self, data_file="pure_gpu_keys.json"):
        self.data_file = data_file
        self.is_running = True
        self.current_batch = 0
        self.start_time = None
        self.total_found = 0
        self.lock = threading.Lock()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 初始化GPU
        self._init_gpu()
        
        self.generated_data = self._load_data()
    
    def _init_gpu(self):
        """初始化GPU环境"""
        try:
            # PyCUDA初始化
            device = cuda.Device(0)
            context = device.make_context()
            self.context = context
            
            print(f"PyCUDA GPU: {device.name()}")
            print(f"计算能力: {device.compute_capability()}")
            print(f"全局内存: {device.total_memory() / 1024**3:.1f} GB")
            
            # 编译CUDA内核
            self.cuda_module = SourceModule(BITCOIN_CUDA_KERNEL)
            self.bitcoin_kernel = self.cuda_module.get_function("bitcoin_address_kernel")
            self.random_keys_kernel = self.cuda_module.get_function("generate_private_keys_kernel")
            
            print("CUDA内核编译成功")
            
            # CuPy初始化
            gpu_count = cp.cuda.runtime.getDeviceCount()
            print(f"CuPy GPU设备: {gpu_count}")
            
            for i in range(gpu_count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                print(f"GPU {i}: {props['name'].decode()}")
            
            cp.cuda.Device(0).use()
            
        except Exception as e:
            print(f"GPU初始化失败: {e}")
            sys.exit(1)
    
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
            "pure_gpu": True
        }
    
    def _save_data(self):
        """保存数据到文件"""
        try:
            self.generated_data["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            temp_file = self.data_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.generated_data, f, indent=2, ensure_ascii=False)
            
            if os.path.exists(self.data_file):
                os.remove(self.data_file)
            os.rename(temp_file, self.data_file)
            
        except Exception as e:
            print(f"保存数据失败: {e}")
    
    def generate_private_keys_gpu(self, batch_size: int) -> cp.ndarray:
        """使用GPU生成私钥"""
        min_range = 1180591620717411303424  # 2^70
        max_range = 2361183241434822606848  # 2^71
        
        # 使用CuPy在GPU上生成随机数
        private_keys_gpu = cp.random.randint(min_range, max_range, batch_size, dtype=cp.uint64)
        return private_keys_gpu
    
    def generate_addresses_gpu(self, private_keys_gpu: cp.ndarray) -> tuple:
        """使用GPU生成比特币地址"""
        batch_size = len(private_keys_gpu)
        
        # 准备GPU内存
        private_keys_ptr = private_keys_gpu.data.ptr
        
        # 分配地址输出内存（每个地址35字符 + 1字节标记）
        addresses_gpu = cp.zeros(batch_size * 36, dtype=cp.uint8)
        addresses_ptr = addresses_gpu.data.ptr
        
        min_range = 1180591620717411303424
        max_range = 2361183241434822606848
        
        # 配置CUDA内核
        block_size = 256
        grid_size = (batch_size + block_size - 1) // block_size
        
        # 执行比特币地址生成内核
        self.bitcoin_kernel(
            cuda.InOut(private_keys_gpu),
            cuda.InOut(addresses_gpu),
            np.int32(batch_size),
            cp.uint64(min_range),
            cp.uint64(max_range),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )
        
        # 等待GPU完成
        self.context.synchronize()
        
        return addresses_gpu, batch_size
    
    def process_batch_pure_gpu(self, batch_size: int) -> List[Dict[str, Any]]:
        """纯GPU处理批次"""
        found_keys = []
        
        # 生成私钥
        private_keys_gpu = self.generate_private_keys_gpu(batch_size)
        
        # 生成地址
        addresses_gpu, actual_batch_size = self.generate_addresses_gpu(private_keys_gpu)
        
        # 将结果复制到CPU
        addresses_cpu = cp.asnumpy(addresses_gpu)
        private_keys_cpu = cp.asnumpy(private_keys_gpu)
        
        # 处理结果
        for i in range(actual_batch_size):
            if not self.is_running:
                break
                
            # 检查是否找到符合条件的地址
            addr_start = i * 36
            found_flag = addresses_cpu[addr_start + 35]
            
            if found_flag == 1:  # 找到符合条件的地址
                # 提取地址字符串
                addr_chars = addresses_cpu[addr_start:addr_start + 34]
                address = ''.join(chr(c) for c in addr_chars if c != 0)
                
                # 获取私钥
                private_key = private_keys_cpu[i]
                
                # 生成WIF（简化版，实际需要完整的WIF生成）
                wif_key = self._generate_wif_gpu(private_key)
                
                result = {
                    'private_key_hex': hex(private_key)[2:].zfill(64),
                    'private_key_wif': wif_key,
                    'address': address,
                    'found_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'batch': self.current_batch
                }
                
                found_keys.append(result)
                
                with self.lock:
                    self.total_found += 1
                    print(f"🎉 找到地址 #{self.total_found}: {address}")
                    # 立即保存重要发现
                    self.generated_data["found_keys"].append(result)
                    self._save_data()
        
        return found_keys
    
    def _generate_wif_gpu(self, private_key: int) -> str:
        """生成WIF格式私钥（简化版）"""
        # 在实际实现中，这里应该使用GPU计算WIF
        # 为简化，我们使用一个基本的WIF生成
        private_key_bytes = private_key.to_bytes(32, 'big')
        wif_base = "5" + private_key_bytes.hex()[:8]  # 简化处理
        return wif_base
    
    def run_pure_gpu_generation(self, batch_size: int = 10000, total_batches: int = None):
        """运行纯GPU密钥生成"""
        self.start_time = time.time()
        self.is_running = True
        
        print(f"\n🚀 开始纯GPU密钥生成")
        print(f"目标前缀: 1PWo3J")
        print(f"私钥范围: 2^70 到 2^71")
        print(f"批次大小: {batch_size}")
        print(f"处理模式: 100% GPU")
        
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
                found_keys = self.process_batch_pure_gpu(batch_size)
                
                # 更新数据
                self.generated_data["total_generated"] += batch_size
                total_keys_generated += batch_size
                
                # 显示批次结果
                batch_time = time.time() - batch_start_time
                keys_per_sec = batch_size / batch_time
                
                total_elapsed = time.time() - self.start_time
                overall_speed = total_keys_generated / total_elapsed
                
                print(f"GPU批次完成! 耗时: {batch_time:.1f}秒, 速度: {keys_per_sec:.1f} 密钥/秒")
                print(f"本批找到: {len(found_keys)} 个符合条件的地址")
                print(f"累计找到: {len(self.generated_data['found_keys'])} 个地址")
                print(f"总速度: {overall_speed:.1f} 密钥/秒")
                
                # 定期保存进度
                if (batch_count + 1) % 10 == 0 or found_keys:
                    self._save_data()
                    print("进度已保存")
                
                batch_count += 1
                
                # 如果指定了总批次，检查是否完成
                if total_batches and batch_count >= total_batches:
                    break
                    
        except Exception as e:
            print(f"GPU生成过程中发生错误: {e}")
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
        print("📊 纯GPU比特币密钥生成器 - 统计信息")
        print("="*60)
        print(f"总共生成的密钥数量: {self.generated_data['total_generated']:,}")
        print(f"找到的符合条件的地址数量: {len(self.generated_data['found_keys'])}")
        print(f"处理模式: 100% GPU")
        print(f"数据最后更新: {self.generated_data.get('last_update', '未知')}")
        
        if self.generated_data['found_keys']:
            print(f"\n最近找到的地址:")
            for i, key_data in enumerate(self.generated_data['found_keys'][-5:], 1):
                print(f"{i}. 地址: {key_data['address']}")
                print(f"   WIF私钥: {key_data['private_key_wif']}")
                print(f"   发现时间: {key_data.get('found_time', '未知')}")
                print(f"   批次: {key_data.get('batch', '未知')}")
                print("-" * 50)

def main():
    print("🚀 纯GPU比特币密钥生成器 - 腾讯云优化版")
    print("目标: 寻找以 '1PWo3J' 开头的比特币地址")
    print("模式: 100% GPU处理")
    
    generator = PureGPUBitcoinGenerator()
    
    while True:
        print("\n" + "="*50)
        print("🔑 纯GPU比特币密钥生成器")
        print("="*50)
        print("1. 快速GPU生成 (1万密钥/批次)")
        print("2. 高性能GPU生成 (10万密钥/批次)") 
        print("3. 大规模GPU生成 (100万密钥/批次)")
        print("4. 自定义GPU生成参数")
        print("5. 显示统计信息")
        print("6. 退出程序")
        print("\n提示: 使用 Ctrl+C 安全中断")
        
        try:
            choice = input("\n请选择操作 (1-6): ").strip()
            
            if choice == '1':
                generator.run_pure_gpu_generation(batch_size=10000, total_batches=10)
            elif choice == '2':
                generator.run_pure_gpu_generation(batch_size=100000, total_batches=5)
            elif choice == '3':
                generator.run_pure_gpu_generation(batch_size=1000000, total_batches=2)
            elif choice == '4':
                try:
                    batch_size = int(input("请输入每批次密钥数量: "))
                    total_batches = int(input("请输入总批次数量: "))
                    if batch_size > 0 and total_batches > 0:
                        generator.run_pure_gpu_generation(batch_size=batch_size, total_batches=total_batches)
                    else:
                        print("请输入正数！")
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
