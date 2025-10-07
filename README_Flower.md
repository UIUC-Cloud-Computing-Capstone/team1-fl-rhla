# 🌸 Flower Federated Learning Implementation

Flower-based federated learning framework for heterogeneous LoRA allocation with multi-core CPU optimization.

## 🚀 Quick Start

### Prerequisites

**Choose ONE option:**

#### Option A: Conda Environment (Recommended)
```bash
conda env create --name env.fl --file=environment.yml
conda activate env.fl
```

#### Option B: Pip Installation
```bash
pip install flwr[simulation] accelerate torch torchvision transformers
```

### Running

**Terminal 1 - Start Server**:
```bash
python flower_server_minimal.py --server_address 0.0.0.0 --server_port 8080 --config_name experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_10_dir-noprior-s50-e50.yaml
```

**Terminal 2 - Start Client**:
```bash
python flower_client_simple.py --server_address localhost --server_port 8080 --client_id 0 --config_name experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_10_dir-noprior-s50-e50.yaml
```

### Multiple Clients

Start additional clients with different `--client_id` values (1, 2, 3, etc.) in separate terminals.

### Multi-Machine Setup

Replace `localhost` with the server's IP address on client machines.

## 🔧 Configuration

**Key Parameters:**
- `--server_address`: Server IP (0.0.0.0 for server, localhost for client)
- `--server_port`: Server port (default: 8080)
- `--client_id`: Unique client identifier
- `--config_name`: Configuration file path

**CPU Optimization:**
```bash
export OMP_NUM_THREADS=8        # Override CPU thread count
export TORCH_NUM_THREADS=8      # PyTorch threads
```

## 🏗️ Architecture

**Components:**
- `flower_server_minimal.py`: Central server (FedAvg strategy)
- `flower_client_simple.py`: Client with simulated training
- `algorithms/solver/fl_utils.py`: Common utilities and aggregation

**Features:**
- Multi-core CPU utilization
- LoRA-aware aggregation
- Heterogeneous client support
- Automatic client management

## 🐛 Troubleshooting

**Common Issues:**
- **Connection Refused**: Start server before clients, check firewall
- **Missing Dependencies**: `pip install -r requirements_flower.txt`
- **Slow Training**: Check CPU usage with `htop`, verify multi-core setup

**Debug Mode:**
```bash
export FLWR_LOG_LEVEL=DEBUG
python flower_server_minimal.py --config_name your_config.yaml
```

## 📚 Resources

- [Flower Documentation](https://flower.dev/docs/)
- [Flower Examples](https://github.com/adap/flower/tree/main/examples)
