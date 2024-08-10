# ISY503 Intelligent Systems - Project

This project involves training a neural network to perform sentiment analysis on a dataset of labeled product reviews, classifying them as positive or negative. The trained model is integrated into a web application for real-time sentiment analysis using FastAPI for the backend and a simple JavaScript page for entering reviews and displaying results.

## Prerequisites

Before you begin, make sure you have the following installed:
- Python 3.7 or higher
- pip (Python package installer)

## Setup and Installation

### 1. Clone the repository:
```bash
git clone https://github.com/sergey-chek/sentiment-analysis-isy503.git
```
```bash
cd sentiment-analysis-isy503
```

### 2. Create a virtual environment:
```bash
python -m venv venv
```

### 3. Activate the virtual environment:
**- Windows:**
```bash
.\venv\Scripts\activate
```
**- MacOS and Linux:**
```bash
source venv/bin/activate
```

### 4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### 1. Start the FastAPI server using Uvicorn:

The following command will start the FastAPI server on `http://127.0.0.1:8000/` by default.
```bash
uvicorn main:app --reload
```
> [!NOTE]
> If you use a different port for the server, make sure to update the `API_ENDPOINT` variable in `static/script.js` to match the correct port.

### 2. Access the frontend

Open your web browser and navigate to `http://127.0.0.1:8000/` (or to your custom endpoint).

## License

This project is licensed under the MIT License.

## Acknowledgments

This project was developed as part of the ISY503 Intelligent Systems course at Torrens University of Australia.
