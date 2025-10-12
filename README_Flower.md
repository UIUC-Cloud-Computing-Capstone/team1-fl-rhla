# 🌸 Flower Federated Learning Framework

Comprehensive Flower-based federated learning with LoRA allocation, multi-core CPU optimization, and automated multi-client deployment.

## 🚀 Quick Start

### Prerequisites
```bash
# Option A: Conda (Recommended)
conda env create --name env_flower.fl --file=environment-flower.yml
conda activate env_flower.fl

# Option B: Pip
pip install flwr[simulation] accelerate torch torchvision transformers
```

### 🔧 Manual Setup (Recommended for development)
```bash
# Terminal 1 - Server
python flower_server.py --server_address 0.0.0.0 --server_port 8080 --config_name experiments/flower/cifar100_vit_lora/fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_20_dir-noprior-s50-e50.yaml --log_level INFO

# Terminal 2 - Client

python flower_client.py --server_address localhost --server_port 8080 --client_id 0 --config_name experiments/flower/cifar100_vit_lora/fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_20_dir-noprior-s50-e50.yaml --log_level INFO

# To force cpu:
python flower_client.py --server_address localhost --server_port 8080 --client_id 0 --config_name experiments/flower/cifar100_vit_lora/fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_20_dir-noprior-s50-e50.yaml --log_level INFO --force_cpu

```

### 🎯 Automated Setup (for experiment)

**Quick Testing:**
```bash
./run_flower_simple.sh          # 3 clients, 5 rounds
./run_flower_simple.sh 5 10     # 5 clients, 10 rounds
```

**Production:**
```bash
./run_flower.sh                 # 10 clients, 10 rounds, INFO logging
./run_flower.sh 5 20 DEBUG      # 5 clients, 20 rounds, DEBUG logging
./run_flower.sh --help          # Show help
```

**Features:**
- ✅ Automatic cleanup & port checking
- ✅ Individual log files per client
- ✅ Real-time monitoring
- ✅ Graceful shutdown with Ctrl+C

## 📊 CPU Utilization

**Each client and server process uses ALL available CPU cores:**
- Separate processes for maximum parallelization
- PyTorch, NumPy, OpenMP configured for all cores
- Total usage = (N clients + 1 server) × CPU cores

**Example (8-core system):**
```
System (8 cores)
├── Server (8 cores)
├── Client 0 (8 cores)
├── Client 1 (8 cores)
└── ... (each client uses all 8 cores)
```

## 🔧 Configuration

**Key Parameters:**
- `--num_rounds`: Training rounds
- `--client_id`: Unique client ID
- `--log_level`: DEBUG, INFO, WARNING, ERROR
- `--config_name`: YAML config file

**Default Config:**
`experiments/flower/cifar100_vit_lora/fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_20_dir-noprior-s50-e50.yaml`

## 🏗️ Architecture

**Components:**
- `flower_server.py`: Central server (FedAvg)
- `flower_client.py`: Client with LoRA support
- `run_flower.sh`: Full-featured deployment script
- `run_flower_simple.sh`: Quick testing script

**Features:**
- Multi-core CPU optimization
- LoRA-aware parameter aggregation
- Automated multi-client deployment
- Real-time monitoring & logging
- Modern Flower API compatibility

## 📁 Logs

**Automated Scripts:**
- Full: `server.log` + `client_logs/client_X.log`
- Simple: Terminal output only

**Manual:** Terminal output only

## 🐛 Troubleshooting

**Common Issues:**
```bash
# Port in use
lsof -i :8080
kill <PID>

# Permission denied
chmod +x run_flower.sh run_flower_simple.sh

# Debug mode
export FLWR_LOG_LEVEL=DEBUG
```

**Required Files:**
- `flower_server.py`
- `flower_client.py`
- `experiments/flower/cifar100_vit_lora/fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_20_dir-noprior-s50-e50.yaml`

## 📊 Example Output

```bash
==========================================
Flower Federated Learning Runner
==========================================
Number of clients: 5
Number of rounds:  10
Log level:         INFO
==========================================

[INFO] Starting Flower server...
[SUCCESS] Server started with PID: 12345
[INFO] Starting 5 Flower clients...
[SUCCESS] All 5 clients started

2025-10-07 20:47:21,860 - Client 0 - INFO - Client 0 starting training for round 1 (epochs=3, lr=0.0100)
2025-10-07 20:47:22,881 - Client 1 - INFO - Client 1 starting training for round 1 (epochs=3, lr=0.0100)
```

## 🌐 Multi-Machine

Replace `localhost` with server IP on client machines.

## 📚 Resources

- [Flower Documentation](https://flower.dev/docs/)
- [Flower Examples](https://github.com/adap/flower/tree/main/examples)

## 🎯 Summary

**This framework provides:**
- Easy deployment with automated scripts
- Maximum CPU utilization across processes
- Production-ready configuration & error handling
- Comprehensive logging & monitoring
- Scalable architecture for various experiment sizes

**Choose automated scripts for easy experiments, or manual setup for fine control.**