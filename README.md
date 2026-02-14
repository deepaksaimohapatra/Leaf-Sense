# Leaf-Sense: Intelligent Plant Disease Detection

Leaf-Sense is a full-stack application designed to help farmers and gardeners identify diseases in crops using deep learning. Currently, it supports high-accuracy detection for **Apple**, **Tomato**, and **Potato** plants.

## 🚀 Features

- **Automated Diagnosis**: Upload a leaf image and get an instant health assessment.
- **Deep Learning Powered**: Utilizes PyTorch and Torchvision models for robust image classification.
- **Modern UI**: A sleek, responsive frontend built with React and Tailwind CSS.
- **Fast API Backend**: High-performance backend services powered by FastAPI.

## 🛠️ Tech Stack

### Backend
- **Core**: Python 3.x, FastAPI
- **Deep Learning**: PyTorch, Torchvision
- **Data Handling**: NumPy, Pillow, Scikit-learn
- **Server**: Uvicorn

### Frontend
- **Framework**: React (Vite)
- **Styling**: Tailwind CSS
- **HTTP Client**: Axios

## 📂 Project Structure

```text
├── api/             # FastAPI backend implementation
├── frontend/        # React (Vite) frontend application
├── src/             # Core logic for preprocessing, training, and inference
│   ├── inference/   # Model prediction logic
│   ├── models/      # Neural network architectures
│   └── preprocessing/ # Image augmentation and cleaning
├── models_saved/    # Trained binary models (Git ignored)
└── data/            # Dataset storage (Git ignored)
```

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.8+
- Node.js & npm

### 2. Backend Setup
```bash
# Navigate to root and create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
python api/main.py
```
The API will be available at `http://localhost:8000`.

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
The UI will be available at `http://localhost:5173`.

## 🧪 Usage

1. Open the frontend in your browser.
2. Select the plant type (Apple, Tomato, or Potato).
3. Upload an image of the affected plant leaf.
4. View the diagnosis result showing whether the plant is healthy or has a specific disease.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
