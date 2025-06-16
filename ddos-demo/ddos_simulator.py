import pandas as pd
import os
import asyncio
import aiohttp
import glob
import random
from collections import Counter, defaultdict

# ====================== PHẦN MAPPING NHÓM NHÃN CHA ======================
attack_group_map = {
    # DrDoS/Reflection
    'DrDoS_DNS': 'DrDoS',
    'DrDoS_SNMP': 'DrDoS',
    'DrDoS_NTP': 'DrDoS',
    'DrDoS_MSSQL': 'DrDoS',
    'DrDoS_SSDP': 'DrDoS',
    'DrDoS_UDP': 'DrDoS',
    'TFTP': 'TFTP',
    # DDoS truyền thống
    'UDP': 'UDP',
    'UDPLag': 'UDP',
    'Syn': 'Syn',
    'MSSQL': 'MSSQL',
    'LDAP': 'LDAP',
    # DoS - đơn lẻ
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'Heartbleed': 'Other',
    # Web
    'Web Attack � Brute Force': 'Web Attack',
    'Web Attack � XSS': 'Web Attack',
    'Web Attack � Sql Injection': 'Web Attack',
    # Brute Force & Infiltration
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Infiltration': 'Other',
    'Bot': 'Other',
    # PortScan
    'PortScan': 'PortScan',
    # NetBIOS
    'NetBIOS': 'Other',

}
def group_attack_type(x):
    if x == 'Benign':
        return 'Benign'
    return attack_group_map.get(x, 'Other')

# ====================== THÔNG SỐ VÀ ĐỌC FILE ======================
DATA_DIR = r'E:\DACS\DACS\Data\2 file'   # Đổi lại đường dẫn nếu cần
FILES = glob.glob(os.path.join(DATA_DIR, '*.parquet'))

features_names = [
    'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packets Length Total', 'Bwd Packets Length Total', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Avg Packet Size',
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init Fwd Win Bytes', 'Init Bwd Win Bytes', 'Fwd Act Data Packets', 'Fwd Seg Size Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
    'Idle Max', 'Idle Min'
]

URL = "http://localhost:8000/predict"
SAMPLE_NUM = 10   # Số lượng sample mỗi file

# ====================== THỐNG KÊ KẾT QUẢ ======================
stats = Counter()
detailed_stats = defaultdict(Counter)

def generate_benign_sample():
    # Giả lập giá trị hợp lý cho mỗi feature, bạn có thể tinh chỉnh lại range cho sát dữ liệu thật hơn
    return {
        name: random.uniform(0, 1)   # Bạn nên thay min/max đúng với từng feature nếu biết
        for name in features_names
    }

async def send_random_benign(session, n=10):
    for _ in range(n):
        payload = generate_benign_sample()
        payload['ip'] = random_ip()
        try:
            async with session.post(URL, data=payload, timeout=30) as resp:
                print(f"Random Benign sent, status: {resp.status}")
        except Exception as e:
            print(f"Lỗi gửi random benign: {e}")


def random_ip():
    return '.'.join(str(random.randint(1, 254)) for _ in range(4))

async def send_row(session, idx, row, true_label, file_name):
    payload = {name: float(row[name]) for name in features_names}
    payload['ip'] = random_ip()  # random mỗi lần 1 IP
    try:
        async with session.post(URL, data=payload, timeout=30) as resp:
            status = resp.status
            if status == 200:
                data = await resp.json()
                stats['benign'] += 1
                detailed_stats[true_label]['Benign'] += 1
                print(f"[{os.path.basename(file_name)}] {payload['ip']} | THỰC: {true_label} | DỰ ĐOÁN: Benign | {data['detection_time']} ms | Status:200")
            elif status == 403:
                detail = await resp.json()
                d = detail.get("detail", {})
                predict_label = d.get('result', '?')
                stats['blocked'] += 1
                detailed_stats[true_label][predict_label] += 1
                print(f"[{os.path.basename(file_name)}] {d.get('ip')} | THỰC: {true_label} | DỰ ĐOÁN: {predict_label} | {d.get('detection_time')} ms | Ngăn chặn: {d.get('block_time')} ms | Status:403")
            elif status == 429:
                stats['429'] += 1
                print(f"[{os.path.basename(file_name)}] {payload['ip']} | 429 Too Many Requests")
            else:
                stats['other_error'] += 1
                print(f"[{os.path.basename(file_name)}] {payload['ip']} | Unexpected {status}")
    except Exception as e:
        stats['send_error'] += 1
        print(f"[{os.path.basename(file_name)}] Lỗi gửi: {e}")

async def main():
    connector = aiohttp.TCPConnector(limit=2)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        idx = 0
        for file_path in FILES:
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                print(f"Lỗi đọc file {file_path}: {e}")
                continue
            for i, row in df.head(SAMPLE_NUM).iterrows():
                # Có thể đổi 'Label' thành 'AttackType' nếu cột bạn lấy là như vậy
                if 'Label' in row:
                    true_label = row['Label']
                elif 'AttackType' in row:
                    true_label = row['AttackType']
                else:
                    true_label = os.path.basename(file_path).replace('-testing.parquet','')
                tasks.append(asyncio.create_task(send_row(session, idx, row, true_label, file_path)))
                idx += 1
                await asyncio.sleep(0.05)   # delay nhỏ giúp tránh nghẽn kết nối
        await asyncio.gather(*tasks)

    # =================== PHÂN TÍCH KẾT QUẢ THEO NHÓM ===================
    print("\n===== TỔNG KẾT =====")
    print(f"Tổng số sample gửi: {sum(stats.values())}")
    print(f"- Thành công (Benign): {stats['benign']}")
    print(f"- Bị chặn/ngăn (Attack): {stats['blocked']}")
    print(f"- Lỗi 429 (Too Many Requests): {stats['429']}")
    print(f"- Lỗi gửi khác: {stats['send_error'] + stats['other_error']}")

    # Gom detailed_stats theo nhóm nhãn cha
    detailed_stats_grouped = defaultdict(Counter)
    for true_label, pred_count in detailed_stats.items():
        group_label = group_attack_type(true_label)
        for pred, cnt in pred_count.items():
            detailed_stats_grouped[group_label][pred] += cnt

    print("\n===== PHÂN TÍCH THEO NHÓM NHÃN CHA (ĐÃ GỘP) =====")
    for group_label, pred_count in detailed_stats_grouped.items():
        total = sum(pred_count.values())
        print(f"Nhóm thật: {group_label}")
        for pred, cnt in pred_count.items():
            print(f"   Dự đoán: {pred} - {cnt}/{total} ({cnt/total:.2%})")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(main())
    