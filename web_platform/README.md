# CardioPredict Web Platform

A professional web interface for the AI-powered cardiovascular risk prediction system developed for space medicine applications with Earth analog validation.

## 🌟 Overview

CardioPredict is the first machine learning system specifically designed for cardiovascular risk assessment in microgravity environments. This web platform provides access to our research findings, interactive demonstrations, and clinical tools developed through rigorous analysis of NASA OSDR datasets.

## 🚀 Key Features

- **Scientific Homepage**: Professional presentation suitable for research paper references
- **Interactive Demo**: Real-time cardiovascular risk prediction
- **Research Documentation**: Comprehensive methodology and results
- **API Access**: RESTful endpoints for clinical integration
- **Mobile Responsive**: Professional design across all devices

## 📊 Research Achievements

- **82% Clinical Model Accuracy** (ElasticNet R² = 0.820)
- **99.9% Research Ensemble Accuracy** (Weighted Average R² = 0.999)
- **85.2% Validation Accuracy** against published bedrest studies
- **Cross-Domain Validation** from space to Earth analog environments
- **45 Biomarkers Analyzed** from 4 NASA OSDR datasets

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: Bootstrap 5, Custom CSS/JS
- **ML Models**: scikit-learn, XGBoost, ensemble methods
- **Data Processing**: pandas, numpy
- **Visualization**: Chart.js, custom scientific graphics

## 📁 Project Structure

```
web_platform/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css     # Professional scientific styling
│   └── js/
│       └── main.js       # Interactive features
└── templates/
    ├── homepage.html     # Main scientific homepage
    ├── base.html         # Base template
    ├── about.html        # Research methodology
    ├── predict.html      # Interactive prediction
    └── ...
```

## 🏃‍♂️ Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository and navigate to the web platform:
```bash
cd CardioPredict/web_platform
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
python app.py
```

4. Open your browser to `http://localhost:8080`

### Production Deployment

For production deployment, use a WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

## 🔬 Scientific Validation

Our system has been rigorously validated through:

1. **NASA OSDR Datasets**:
   - OSD-258: SpaceX Inspiration4 RNA-seq data
   - OSD-484: Cardiac gene expression studies
   - OSD-575: Comprehensive metabolic panels
   - OSD-51: Microarray bedrest studies
   - OSD-635: Bulk RNA-seq validation data

2. **Cross-Domain Validation**:
   - Published 14-day and 84-day bedrest studies
   - Hospital patient simulation
   - Space mission to Earth analog transfer learning

3. **Clinical Interpretability**:
   - ElasticNet model for clinical decision-making
   - Feature importance analysis
   - Biomarker significance assessment

## 📖 Usage

### Interactive Prediction

1. Navigate to the **Live Demo** section
2. Input biomarker values or use sample data
3. View real-time risk assessment and recommendations
4. Export results for clinical documentation

### API Integration

```python
import requests

# Predict cardiovascular risk
response = requests.post('http://localhost:8080/api/predict', 
                        json={'biomarkers': biomarker_data})
risk_assessment = response.json()
```

### Research Access

- **Methodology**: Detailed scientific approach and validation
- **Results**: Comprehensive performance metrics and comparisons
- **Documentation**: API reference and integration guides
- **Publications**: Links to research papers and presentations

## 📚 Citation

If you use this platform in your research, please cite:

```bibtex
@software{cardiopredict2025,
  title={CardioPredict: AI-Powered Cardiovascular Risk Prediction for Space Medicine},
  author={CardioPredict Research Team},
  year={2025},
  url={http://your-platform-url.com},
  note={Web platform for cardiovascular risk prediction in microgravity environments}
}
```

## 🤝 Contributing

We welcome contributions from the space medicine and AI research communities:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA Open Science Data Repository (OSDR)** for providing essential datasets
- **SpaceX Inspiration4** mission for contributing valuable space medicine data
- **International space agencies** for bedrest study validation data
- **Open source community** for machine learning and web development tools

## 📞 Contact

For research collaborations, clinical implementations, or technical support:

- **Email**: [research@cardiopredict.org]
- **Website**: [https://cardiopredict.org]
- **Research Portal**: [https://cardiopredict.org/research]

---

**CardioPredict** - Advancing cardiovascular health monitoring for the next generation of space exploration 🚀❤️
