# SAR Drone AI System

An advanced Search and Rescue drone system with integrated AI capabilities for autonomous mission planning, safety analysis, and real-time decision making.

## 🚁 Features

- **AI-Powered Mission Planning**: Intelligent route optimization and search pattern generation
- **Real-time Safety Analysis**: Continuous monitoring and risk assessment
- **Multi-Drone Coordination**: Swarm intelligence for coordinated search operations
- **Ollama Integration**: Local AI models for offline operation
- **Production-Grade Reliability**: Comprehensive error handling and fallback systems

## 🏗️ Architecture

```
app/
├── api/                 # REST API endpoints
├── core/               # Core system components
│   ├── ai_manager.py   # AI model management
│   ├── config.py       # Configuration management
│   ├── database.py     # Database operations
│   └── errors.py       # Error handling
├── modules/            # AI modules
│   ├── ai_coordinator.py
│   ├── ai_location_analyzer.py
│   ├── ai_mission_interpreter.py
│   ├── ai_planner.py
│   └── simulator.py
├── data/               # Data storage
├── logs/               # System logs
└── main.py            # Application entry point
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Ollama (for local AI models)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sar-drone-ai.git
   cd sar-drone-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ollama**
   ```bash
   # Install required AI models
   python download_all_models.py
   ```

4. **Configure the system**
   ```bash
   # Copy and edit configuration
   cp config.example.py core/config.py
   ```

5. **Run the system**
   ```bash
   python main.py
   ```

## 🤖 AI Models

The system uses multiple AI models for different purposes:

- **mistral:7b-instruct**: Mission planning and complex analysis
- **phi:2.7b**: Fast tactical decisions
- **llama2:7b**: Safety and risk assessment

## 📊 Monitoring

The system provides comprehensive monitoring:
- Performance metrics
- AI response caching
- Health monitoring
- Safety event logging

## 🔧 Configuration

Key configuration options in `core/config.py`:
- Ollama host settings
- Database configuration
- Logging levels
- Safety thresholds

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Safety Notice

This is a research system. Always test in controlled environments and follow local drone regulations.

## 🔄 Auto-Update Setup

This repository is configured for automatic updates every few minutes. See the GitHub Actions workflow for details. 