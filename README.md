# Graph Mentor - Hackathon Project 🚀

Graph Mentor is a tool designed to help individuals identify, develop, and track their personal and professional competencies through structured action plans.

## 🛠 Installation & Setup

Follow these steps to set up and run the project locally.

### 1️⃣ Clone the repository
Open a terminal and run:
```bash
git clone https://github.com/tomas-pucutay/arangodb_graph_mentor_hackaton.git
```

### 2️⃣ Navigate into the project folder
```bash
cd arangodb_graph_mentor_hackaton
```

### 3️⃣ Set up the Python path
Ensure your environment recognizes the project path:
```bash
export PYTHONPATH=$(pwd)
```

### 4️⃣ Configure environment variables
Rename the environment configuration file:
```bash
cp .copy.env .env
Edit .env with your specific configuration settings if needed.
```

### 5️⃣ Install dependencies
Make sure you have Python installed, then install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate # .venv/Scripts/activate for Windows
pip install -r requirements.txt
```

### 6️⃣ Run the application
Start the application using Streamlit:

```bash
streamlit run main.py
```