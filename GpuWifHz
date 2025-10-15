import hashlib
import base58
import ecdsa
from ecdsa.curves import SECP256k1
import time
import logging
import signal
import sys
import os
from datetime import datetime
import threading
from queue import Queue
import struct

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bitcoin_gpu_miner.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 尝试导入PyCUDA
try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
    import numpy as np
    GPU_AVAILABLE = True
    logger.info("PyCUDA 可用，启用 GPU 加速")
except ImportError as e:
    GPU_AVAILABLE = False
    logger.warning(f"PyCUDA 不可用: {e}，回退到 CPU 模式")

class BitcoinGPUKeyGenerator:
    def __init__(self, start_wif, target_suffix, batch_size=50000):
        self.start_wif = start_wif
        self.target_suffix = target_suffix
        self.batch_size = batch_size
        self.found_count = 0
        self.total_checked = 0
        self.start_time = time.time()
        self.should_stop = False
        
        # 信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 初始化起始私钥
        self.current_private_key = self.wif_to_private_key(start_wif)
        logger.info(f"起始私钥(hex): {self.current_private_key}")
        
        # 验证起始地址
        start_address = self.private_key_to_address(self.current_private_key)
        logger.info(f"起始地址: {start_address}")
        
        # GPU初始化
        if GPU_AVAILABLE:
            self.setup_gpu()
        
        # 创建结果目录
        os.makedirs('results', exist_ok=True)
    
    def signal_handler(self, signum, frame):
        """处理中断信号"""
        logger.info(f"收到信号 {signum}，准备优雅退出...")
        self.should_stop = True
    
    def wif_to_private_key(self, wif_key):
        """将WIF格式私钥转换回原始私钥"""
        decoded = base58.b58decode(wif_key)
        private_key_hex = decoded[1:33].hex()
        return private_key_hex
    
    def private_key_to_wif(self, private_key_hex):
        """将私钥转换为WIF格式"""
        extended_key = '80' + private_key_hex + '01'
        first_sha = hashlib.sha256(bytes.fromhex(extended_key)).digest()
        second_sha = hashlib.sha256(first_sha).digest()
        checksum = second_sha[:4]
        final_key = extended_key + checksum.hex()
        return base58.b58encode(bytes.fromhex(final_key)).decode('ascii')
    
    def setup_gpu(self):
        """设置GPU环境和编译CUDA内核"""
        try:
            # 获取GPU信息
            device = cuda.Context.get_device()
            logger.info(f"使用GPU: {device.name()}")
            logger.info(f"GPU计算能力: {device.compute_capability()}")
            logger.info(f"GPU内存: {device.total_memory() // (1024**3)} GB")
            
            # 编译CUDA内核用于哈希计算
            self.compile_cuda_kernels()
            
        except Exception as e:
            logger.error(f"GPU初始化失败: {e}")
            global GPU_AVAILABLE
            GPU_AVAILABLE = False
    
    def compile_cuda_kernels(self):
        """编译CUDA内核"""
        try:
            # 简单的CUDA内核示例 - 实际应用中需要更复杂的内核
            cuda_source = """
            __global__ void increment_keys(unsigned long long* keys, int batch_size, unsigned long long start) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < batch_size) {
                    keys[idx] = start + idx;
                }
            }
            """
            
            mod = SourceModule(cuda_source)
            self.increment_kernel = mod.get_function("increment_keys")
            
            logger.info("CUDA内核编译成功")
            
        except Exception as e:
            logger.error(f"CUDA内核编译失败: {e}")
            GPU_AVAILABLE = False
    
    def generate_private_keys_batch_gpu(self, start_key_hex, count):
        """使用GPU生成一批私钥"""
        if not GPU_AVAILABLE:
            return self.generate_private_keys_batch_cpu(start_key_hex, count)
        
        try:
            start_int = int(start_key_hex, 16)
            
            # 分配GPU内存
            keys_gpu = gpuarray.zeros(count, dtype=np.uint64)
            
            # 配置GPU网格和块大小
            block_size = 256
            grid_size = (count + block_size - 1) // block_size
            
            # 执行内核
            self.increment_kernel(keys_gpu, np.int32(count), np.uint64(start_int),
                                block=(block_size, 1, 1), grid=(grid_size, 1))
            
            # 将结果复制回CPU
            keys_cpu = keys_gpu.get()
            
            # 转换为十六进制
            keys_hex = [format(key, '064x') for key in keys_cpu]
            
            return keys_hex
            
        except Exception as e:
            logger.error(f"GPU私钥生成失败: {e}，回退到CPU")
            return self.generate_private_keys_batch_cpu(start_key_hex, count)
    
    def generate_private_keys_batch_cpu(self, start_key_hex, count):
        """CPU版本的私钥批量生成"""
        start_int = int(start_key_hex, 16)
        max_private_key = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140
        
        keys_hex = []
        current_int = start_int
        
        for i in range(count):
            if current_int > max_private_key:
                break
            keys_hex.append(format(current_int, '064x'))
            current_int += 1
        
        return keys_hex
    
    def private_key_to_address(self, private_key_hex):
        """从私钥生成比特币地址"""
        try:
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
        except Exception as e:
            logger.error(f"地址生成错误: {e}")
            return None
    
    def process_batch_parallel(self, private_keys_batch):
        """并行处理一批私钥"""
        matches = []
        
        # 使用多线程并行处理
        num_threads = min(os.cpu_count(), len(private_keys_batch) // 100 + 1)
        chunk_size = max(1, len(private_keys_batch) // num_threads)
        
        results_queue = Queue()
        threads = []
        
        for i in range(num_threads):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_threads - 1 else len(private_keys_batch)
            chunk = private_keys_batch[start_idx:end_idx]
            
            thread = threading.Thread(
                target=self.process_chunk,
                args=(chunk, results_queue, i)
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 收集结果
        while not results_queue.empty():
            chunk_matches = results_queue.get()
            matches.extend(chunk_matches)
        
        return matches, len(private_keys_batch)
    
    def process_chunk(self, private_keys_chunk, results_queue, thread_id):
        """处理一个私钥块"""
        chunk_matches = []
        
        for private_key_hex in private_keys_chunk:
            if self.should_stop:
                break
                
            address = self.private_key_to_address(private_key_hex)
            if address and address.endswith(self.target_suffix):
                wif = self.private_key_to_wif(private_key_hex)
                chunk_matches.append({
                    'private_key_wif': wif,
                    'address': address,
                    'thread_id': thread_id
                })
        
        results_queue.put(chunk_matches)
    
    def save_result(self, result):
        """保存找到的结果"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename = f"results/found_addresses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"时间: {timestamp}\n")
            f.write(f"私钥(WIF): {result['private_key_wif']}\n")
            f.write(f"地址: {result['address']}\n")
            f.write(f"总检查数: {self.total_checked:,}\n")
            f.write("-" * 70 + "\n")
        
        # 同时保存到主文件
        with open('found_addresses.txt', 'a', encoding='utf-8') as f:
            f.write(f"时间: {timestamp}\n")
            f.write(f"私钥(WIF): {result['private_key_wif']}\n")
            f.write(f"地址: {result['address']}\n")
            f.write(f"总检查数: {self.total_checked:,}\n")
            f.write("-" * 70 + "\n")
    
    def print_stats(self):
        """打印统计信息"""
        elapsed_time = time.time() - self.start_time
        keys_per_second = self.total_checked / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(f"已检查: {self.total_checked:,} 个私钥 | "
                   f"找到: {self.found_count} 个匹配 | "
                   f"速度: {keys_per_second:,.0f} 密钥/秒 | "
                   f"运行时间: {elapsed_time:.0f} 秒")
    
    def run(self):
        """主运行循环"""
        logger.info("开始搜索...")
        logger.info(f"目标地址后缀: {self.target_suffix}")
        logger.info(f"批处理大小: {self.batch_size}")
        logger.info(f"GPU加速: {GPU_AVAILABLE}")
        logger.info("按 Ctrl+C 停止搜索")
        print("-" * 80)
        
        last_stats_time = time.time()
        stats_interval = 10  # 每10秒打印一次统计
        
        while not self.should_stop:
            # 生成一批私钥
            if GPU_AVAILABLE:
                private_keys_batch = self.generate_private_keys_batch_gpu(
                    self.current_private_key, self.batch_size
                )
            else:
                private_keys_batch = self.generate_private_keys_batch_cpu(
                    self.current_private_key, self.batch_size
                )
            
            if not private_keys_batch:
                logger.info("私钥空间已耗尽")
                break
            
            # 处理批处理
            matches, checked_in_batch = self.process_batch_parallel(private_keys_batch)
            
            # 更新统计
            self.total_checked += checked_in_batch
            self.found_count += len(matches)
            
            # 保存找到的结果
            for match in matches:
                logger.info(f"🎯 找到匹配地址: {match['address']}")
                self.save_result(match)
            
            # 更新当前私钥
            if private_keys_batch:
                last_key_int = int(private_keys_batch[-1], 16)
                self.current_private_key = format(last_key_int + 1, '064x')
            
            # 定期打印统计
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                self.print_stats()
                last_stats_time = current_time
            
            # 检查是否达到私钥上限
            last_key_int = int(private_keys_batch[-1], 16)
            if last_key_int >= 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140:
                logger.info("已达到私钥空间上限")
                break
        
        # 最终统计
        self.print_stats()
        logger.info("搜索完成")

def main():
    # 配置参数
    start_wif = "KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3rEgBrU3EPcC3MPqMCobk"
    target_suffix = "sVzXU"
    
    # 根据GPU可用性调整批处理大小
    if GPU_AVAILABLE:
        batch_size = 100000  # GPU可以处理更大的批处理
    else:
        batch_size = 50000   # CPU模式使用较小的批处理
    
    # 创建生成器并运行
    generator = BitcoinGPUKeyGenerator(start_wif, target_suffix, batch_size)
    generator.run()

if __name__ == "__main__":
    main()
