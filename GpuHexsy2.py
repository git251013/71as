import os
import json
import sys
import time
import signal
import threading
import atexit
import traceback
import subprocess
import tempfile
from typing import List, Dict, Any
import numpy as np

# 设置环境变量，强制使用 nvcc
os.environ["CUDA_NVCC_EXECUTABLE"] = "/usr/local/cuda/bin/nvcc"

# 依赖检查 - 只安装必要的GPU库
def install_gpu_dependencies():
    """安装GPU相关依赖包"""
    dependencies = ["cupy-cuda11x", "base58", "ecdsa"]
    
    for dep in dependencies:
        try:
            if dep.startswith("cupy"):
                import cupy
                print(f"✓ {dep} 已安装")
            elif dep == "base58":
                import base58
                print(f"✓ {dep} 已安装")
            elif dep == "ecdsa":
                import ecdsa
                print(f"✓ {dep} 已安装")
        except ImportError:
            print(f"正在安装 {dep}...")
            os.system(f"{sys.executable} -m pip install {dep} -i https://pypi.tuna.tsinghua.edu.cn/simple")
            try:
                if dep.startswith("cupy"):
                    import cupy
                elif dep == "base58":
                    import base58
                elif dep == "ecdsa":
                    import ecdsa
                print(f"✓ {dep} 安装成功")
            except ImportError:
                print(f"✗ {dep} 安装失败")

# 安装依赖
install_gpu_dependencies()

# 导入主库
import cupy as cp
import base58
import ecdsa

class NVCUBitcoinGenerator:
    def __init__(self, data_file="nvcc_gpu_keys.json"):
        self.data_file = data_file
        self.is_running = True
        self.current_batch = 0
        self.start_time = None
        self.total_found = 0
        self.lock = threading.Lock()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 注册退出清理函数
        atexit.register(self.cleanup)
        
        # 初始化GPU
        self._init_gpu()
        
        self.generated_data = self._load_data()
    
    def _init_gpu(self):
        """初始化GPU环境"""
        try:
            # CuPy初始化
            gpu_count = cp.cuda.runtime.getDeviceCount()
            print(f"检测到 {gpu_count} 个GPU设备")
            
            for i in range(gpu_count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                print(f"GPU {i}: {props['name'].decode()}")
                print(f"  计算能力: {props['major']}.{props['minor']}")
                print(f"  全局内存: {props['totalGlobalMem'] / 1024**3:.1f} GB")
            
            cp.cuda.Device(0).use()
            
            # 编译CUDA内核
            self._compile_cuda_kernel()
            
        except Exception as e:
            print(f"GPU初始化失败: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def _compile_cuda_kernel(self):
        """使用nvcc编译CUDA内核"""
        try:
            # 创建CUDA内核源代码
            cuda_source = """
#include <cstdint>

extern "C" {
__global__ void generate_keys_and_check(
    uint64_t *keys_output,
    uint8_t *matches_output,
    int num_keys,
    uint64_t min_val,
    uint64_t max_val,
    uint64_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;
    
    // 使用线程ID和种子生成随机私钥
    uint64_t private_key = min_val + (
        (seed * 6364136223846793005ULL + idx * 2862933555777941757ULL) % (max_val - min_val)
    );
    
    // 存储私钥
    keys_output[idx] = private_key;
    
    // 简化的地址匹配逻辑
    // 在实际实现中，这里应该进行完整的地址生成和检查
    uint64_t hash = private_key;
    for (int i = 0; i < 5; i++) {
        hash = (hash * 6364136223846793005ULL + 1442695040888963407ULL);
    }
    
    // 基于哈希值的简单匹配检查
    // 实际应该检查地址是否以"1PWo3J"开头
    uint8_t matches = 0;
    if ((hash & 0xFFFFFF) == 0x1PWo3J) {  // 简化的匹配条件
        matches = 1;
    }
    
    matches_output[idx] = matches;
}

__global__ void sha256_kernel(
    const uint8_t *input,
    uint8_t *output,
    int num_blocks
) {
    // 简化的SHA256实现
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;
    
    // 实际应实现完整的SHA256
    for (int i = 0; i < 32; i++) {
        output[idx * 32 + i] = input[idx * 64 + (i % 64)] ^ i;
    }
}
}
"""
            
            # 创建临时文件保存CUDA源代码
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(cuda_source)
                cuda_file = f.name
            
            # 编译CUDA内核
            cubin_file = cuda_file.replace('.cu', '.cubin')
            
            # 使用nvcc编译
            nvcc_cmd = [
                "nvcc",
                "-cubin",
                "-arch=sm_60",  # 指定计算能力
                "-o", cubin_file,
                cuda_file
            ]
            
            print(f"编译CUDA内核: {' '.join(nvcc_cmd)}")
            result = subprocess.run(nvcc_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"NVCC编译失败: {result.stderr}")
                raise Exception("CUDA内核编译失败")
            
            # 加载编译后的内核
            with open(cubin_file, 'rb') as f:
                cubin = f.read()
            
            # 使用CuPy加载内核
            self.keygen_module = cp.RawModule(code=cubin)
            self.keygen_kernel = self.keygen_module.get_function("generate_keys_and_check")
            self.sha256_kernel = self.keygen_module.get_function("sha256_kernel")
            
            print("CUDA内核编译并加载成功")
            
            # 清理临时文件
            os.unlink(cuda_file)
            os.unlink(cubin_file)
            
        except Exception as e:
            print(f"CUDA内核编译失败: {e}")
            print("将使用纯CuPy实现")
    
    def cleanup(self):
        """清理GPU资源"""
        try:
            # 清理CuPy内存池
            if 'cp' in sys.modules:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            
        except Exception as e:
            print(f"清理GPU资源时出错: {e}")
    
    def signal_handler(self, signum, frame):
        """处理中断信号"""
        print(f"\n收到中断信号，正在保存数据...")
        self.is_running = False
        self._save_data()
        self.cleanup()
        print("数据已保存，GPU资源已清理，退出程序")
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
            "nvcc_compiled": True
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
    
    def generate_keys_gpu(self, batch_size: int) -> tuple:
        """使用GPU生成私钥并初步筛选"""
        min_range = 1180591620717411303424  # 2^70
        max_range = 2361183241434822606848  # 2^71
        
        try:
            # 分配GPU内存
            keys_gpu = cp.zeros(batch_size, dtype=cp.uint64)
            matches_gpu = cp.zeros(batch_size, dtype=cp.uint8)
            
            # 配置内核参数
            block_size = 256
            grid_size = (batch_size + block_size - 1) // block_size
            seed = cp.uint64(int(time.time() * 1000000))
            
            # 执行内核
            self.keygen_kernel(
                (grid_size, 1), (block_size, 1, 1),
                (keys_gpu, matches_gpu, cp.int32(batch_size), 
                 cp.uint64(min_range), cp.uint64(max_range), seed)
            )
            
            # 等待GPU完成
            cp.cuda.stream.get_current_stream().synchronize()
            
            return keys_gpu, matches_gpu
            
        except Exception as e:
            print(f"GPU密钥生成失败: {e}")
            # 回退到纯CuPy实现
            return self._generate_keys_fallback(batch_size)
    
    def _generate_keys_fallback(self, batch_size: int) -> tuple:
        """回退方法 - 使用纯CuPy生成私钥"""
        min_range = 1180591620717411303424
        max_range = 2361183241434822606848
        
        # 使用CuPy生成随机私钥
        keys_gpu = cp.random.randint(min_range, max_range, batch_size, dtype=cp.uint64)
        # 所有匹配标志设为0（在CPU阶段进行完整检查）
        matches_gpu = cp.zeros(batch_size, dtype=cp.uint8)
        
        return keys_gpu, matches_gpu
    
    def process_batch_gpu(self, batch_size: int) -> List[Dict[str, Any]]:
        """使用GPU处理批次"""
        found_keys = []
        
        try:
            # 生成私钥和初步匹配结果
            keys_gpu, matches_gpu = self.generate_keys_gpu(batch_size)
            
            # 将结果复制到CPU
            keys_cpu = cp.asnumpy(keys_gpu)
            matches_cpu = cp.asnumpy(matches_gpu)
            
            # 处理结果
            for i in range(batch_size):
                if not self.is_running:
                    break
                
                private_key = keys_cpu[i]
                
                # 如果GPU初步匹配或需要完整检查
                if matches_cpu[i] == 1 or True:  # 暂时对所有密钥进行完整检查
                    # 生成完整的比特币地址
                    address = self._generate_address_cpu(private_key)
                    
                    if address and address.startswith('1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
'):
                        wif_key = self._generate_wif_cpu(private_key)
                        
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
            
            # 清理GPU内存
            del keys_gpu
            del matches_gpu
            
        except Exception as e:
            print(f"GPU处理批次时出错: {e}")
            traceback.print_exc()
        
        return found_keys
    
    def _generate_address_cpu(self, private_key: int) -> str:
        """在CPU上生成比特币地址"""
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
            
            # SHA256哈希
            import hashlib
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
    
    def _generate_wif_cpu(self, private_key: int) -> str:
        """在CPU上生成WIF格式私钥"""
        try:
            import hashlib
            
            # 添加前缀0x80（主网）和压缩标志0x01
            extended_key = b'\x80' + private_key.to_bytes(32, 'big') + b'\x01'
            
            # 双重SHA256哈希
            first_hash = hashlib.sha256(extended_key).digest()
            second_hash = hashlib.sha256(first_hash).digest()
            
            # 添加校验和
            checksum = second_hash[:4]
            final_key = extended_key + checksum
            
            # Base58编码
            wif = base58.b58encode(final_key).decode('ascii')
            return wif
            
        except Exception as e:
            print(f"WIF生成错误: {e}")
            return None
    
    def run_gpu_generation(self, batch_size: int = 10000, total_batches: int = None):
        """运行GPU密钥生成"""
        self.start_time = time.time()
        self.is_running = True
        
        print(f"\n🚀 开始NVCC编译的GPU密钥生成")
        print(f"目标前缀: 1PWo3J")
        print(f"私钥范围: 2^70 到 2^71")
        print(f"批次大小: {batch_size}")
        print(f"编译方式: NVCC")
        
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
                found_keys = self.process_batch_gpu(batch_size)
                
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
            traceback.print_exc()
        except KeyboardInterrupt:
            print("\n用户中断生成过程")
        finally:
            # 最终保存
            self._save_data()
            total_elapsed = time.time() - self.start_time
            print(f"\n任务完成! 总运行时间: {total_elapsed:.1f} 秒")
            print(f"总生成密钥: {total_keys_generated:,}")
            if total_elapsed > 0:
                print(f"平均速度: {overall_speed:.1f} 密钥/秒")
            print(f"找到地址总数: {len(self.generated_data['found_keys'])}")
    
    def show_statistics(self):
        """显示统计信息"""
        print("\n" + "="*60)
        print("📊 NVCC GPU比特币密钥生成器 - 统计信息")
        print("="*60)
        print(f"总共生成的密钥数量: {self.generated_data['total_generated']:,}")
        print(f"找到的符合条件的地址数量: {len(self.generated_data['found_keys'])}")
        print(f"编译方式: NVCC")
        print(f"数据最后更新: {self.generated_data.get('last_update', '未知')}")
        
        if self.generated_data['found_keys']:
            print(f"\n最近找到的地址:")
            for i, key_data in enumerate(self.generated_data['found_keys'][-5:], 1):
                print(f"{i}. 地址: {key_data['address']}")
                print(f"   WIF私钥: {key_data['private_key_wif']}")
                print(f"   发现时间: {key_data.get('found_time', '未知')}")
                print(f"   批次: {key_data.get('batch', '未知')}")
                print("-" * 50)
    
    def __del__(self):
        """析构函数 - 确保资源清理"""
        self.cleanup()

def main():
    print("🚀 NVCC编译的GPU比特币密钥生成器 - 腾讯云优化版")
    print("目标: 寻找以 '1PWo3J' 开头的比特币地址")
    print("编译方式: NVCC (避免GCC版本依赖)")
    
    generator = None
    try:
        generator = NVCUBitcoinGenerator()
        
        while True:
            print("\n" + "="*50)
            print("🔑 NVCC GPU比特币密钥生成器")
            print("="*50)
            print("1. 快速生成 (1万密钥/批次)")
            print("2. 高性能生成 (10万密钥/批次)") 
            print("3. 大规模生成 (50万密钥/批次)")
            print("4. 自定义生成参数")
            print("5. 显示统计信息")
            print("6. 退出程序")
            print("\n提示: 使用 Ctrl+C 安全中断")
            
            try:
                choice = input("\n请选择操作 (1-6): ").strip()
                
                if choice == '1':
                    generator.run_gpu_generation(batch_size=10000, total_batches=10)
                elif choice == '2':
                    generator.run_gpu_generation(batch_size=100000, total_batches=5)
                elif choice == '3':
                    generator.run_gpu_generation(batch_size=500000, total_batches=2)
                elif choice == '4':
                    try:
                        batch_size = int(input("请输入每批次密钥数量: "))
                        total_batches = int(input("请输入总批次数量: "))
                        if batch_size > 0 and total_batches > 0:
                            generator.run_gpu_generation(batch_size=batch_size, total_batches=total_batches)
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
                
    except Exception as e:
        print(f"程序运行出错: {e}")
        traceback.print_exc()
    finally:
        if generator:
            generator.cleanup()

if __name__ == "__main__":
    main()
