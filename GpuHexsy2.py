import cupy as cp
import hashlib
import base58
import time

class OptimizedBitcoinCollisionGPU:
    def __init__(self, target_address: str):
        self.target_address = target_address
        self.found_key = None
        
        # secp256k1参数
        self.p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
        self.a = 0
        self.b = 7
        
    def _mod_inverse(self, a, modulus):
        """模逆运算"""
        return pow(a, modulus-2, modulus)
    
    def _point_add(self, P1, P2):
        """椭圆曲线点加法"""
        if P1 is None:
            return P2
        if P2 is None:
            return P1
            
        x1, y1 = P1
        x2, y2 = P2
        
        if x1 == x2:
            if y1 == y2:
                # 点加倍
                s = (3 * x1 * x1 + self.a) * self._mod_inverse(2 * y1, self.p) % self.p
            else:
                return None  # 点互为逆元
        else:
            s = (y2 - y1) * self._mod_inverse(x2 - x1, self.p) % self.p
            
        x3 = (s * s - x1 - x2) % self.p
        y3 = (s * (x1 - x3) - y1) % self.p
        
        return (x3, y3)
    
    def _scalar_multiply(self, k, P):
        """标量乘法 k * P"""
        if k == 0:
            return None
            
        # 二进制展开法
        result = None
        addend = P
        
        while k:
            if k & 1:
                result = self._point_add(result, addend)
            addend = self._point_add(addend, addend)
            k >>= 1
            
        return result
    
    def search_single_batch(self, start_key, batch_size):
        """搜索单个批次"""
        print(f"搜索批次: {start_key} 到 {start_key + batch_size}")
        
        # 生成私钥范围
        private_keys = cp.arange(start_key, start_key + batch_size, dtype=cp.uint64)
        
        # 基础点 G
        G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
             0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
        
        found_key = None
        
        for i in range(batch_size):
            priv_key = int(private_keys[i])
            
            # 计算公钥
            public_key = self._scalar_multiply(priv_key, G)
            
            if public_key:
                # 转换为地址并检查
                address = self._public_key_to_address_single(public_key)
                if address == self.target_address:
                    found_key = priv_key
                    break
        
        return found_key
    
    def _public_key_to_address_single(self, public_key):
        """单个公钥到地址的转换"""
        x, y = public_key
        
        # 压缩公钥格式
        prefix = b'\x02' if y % 2 == 0 else b'\x03'
        pub_key_bytes = prefix + x.to_bytes(32, 'big')
        
        # SHA256
        sha256_hash = hashlib.sha256(pub_key_bytes).digest()
        
        # RIPEMD160
        ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
        
        # 添加版本字节
        version_ripemd160 = b'\x00' + ripemd160_hash
        
        # 校验和
        checksum = hashlib.sha256(hashlib.sha256(version_ripemd160).digest()).digest()[:4]
        
        # Base58编码
        binary_address = version_ripemd160 + checksum
        address = base58.b58encode(binary_address).decode('utf-8')
        
        return address

def main():
    TARGET_ADDRESS = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"
    
    print("启动优化版比特币地址碰撞搜索")
    print(f"目标地址: {TARGET_ADDRESS}")
    print(f"搜索范围: 2^70 到 2^71")
    
    # 初始化搜索器
    searcher = OptimizedBitcoinCollisionGPU(TARGET_ADDRESS)
    
    # 搜索参数
    start_range = 2**70
    end_range = 2**71
    batch_size = 10000
    
    current_start = start_range
    keys_checked = 0
    start_time = time.time()
    
    try:
        while current_start < end_range:
            found_key = searcher.search_single_batch(current_start, batch_size)
            
            if found_key:
                print(f"\n🎉 找到匹配的私钥!")
                print(f"私钥 (十六进制): {hex(found_key)}")
                print(f"私钥 (十进制): {found_key}")
                
                # 保存结果
                with open(f"found_key_{int(time.time())}.txt", "w") as f:
                    f.write(f"目标地址: {TARGET_ADDRESS}\n")
                    f.write(f"私钥: {hex(found_key)}\n")
                    f.write(f"找到时间: {time.ctime()}\n")
                break
            
            keys_checked += batch_size
            current_start += batch_size
            
            # 进度报告
            if keys_checked % 100000 == 0:
                elapsed = time.time() - start_time
                rate = keys_checked / elapsed
                print(f"已检查: {keys_checked} 密钥, 速度: {rate:.2f} 密钥/秒")
                
    except KeyboardInterrupt:
        print("\n搜索被用户中断")
    
    print(f"总共检查了 {keys_checked} 个密钥")

if __name__ == "__main__":
    main()
