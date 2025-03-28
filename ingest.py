from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = "Data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()

# Sample financial data (in INR Crores)
years = ["2020", "2021", "2022", "2023"]

# Revenue Data (in crores)
tcs_revenue = [156949, 164177, 191754, 226000]
reliance_revenue = [659205, 499282, 792756, 974864]
hdfc_revenue = [105161, 116177, 132332, 157325]

# Net Profit Data (in crores)
tcs_profit = [32340, 33080, 38327, 42500]
reliance_profit = [39354, 53156, 67000, 74000]
hdfc_profit = [21292, 24356, 28764, 32500]

# Market Capitalization (in lakh crores)
tcs_market_cap = [9.5, 11.2, 12.8, 13.6]
reliance_market_cap = [14.5, 16.2, 17.8, 19.1]
hdfc_market_cap = [7.2, 8.5, 9.8, 11.0]

x = np.arange(len(years))  # Label locations
width = 0.25  # Bar width

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Set a common title
fig.suptitle("Informative Graphs About All Three Companies", fontsize=16, fontweight='bold')

# --- Revenue Graph ---
axes[0].bar(x - width, tcs_revenue, width, label="TCS", color='blue')
axes[0].bar(x, reliance_revenue, width, label="Reliance", color='green')
axes[0].bar(x + width, hdfc_revenue, width, label="HDFC", color='red')
axes[0].set_title("Annual Revenue Comparison (INR Crores)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(years)
axes[0].set_ylabel("Revenue (Crores)")
axes[0].legend()

# --- Net Profit Graph ---
axes[1].bar(x - width, tcs_profit, width, label="TCS", color='blue')
axes[1].bar(x, reliance_profit, width, label="Reliance", color='green')
axes[1].bar(x + width, hdfc_profit, width, label="HDFC", color='red')
axes[1].set_title("Annual Net Profit Comparison (INR Crores)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(years)
axes[1].set_ylabel("Net Profit (Crores)")
axes[1].legend()

# --- Market Capitalization Growth ---
axes[2].plot(years, tcs_market_cap, marker='o', linestyle='-', label="TCS", color='blue')
axes[2].plot(years, reliance_market_cap, marker='s', linestyle='-', label="Reliance", color='green')
axes[2].plot(years, hdfc_market_cap, marker='^', linestyle='-', label="HDFC", color='red')
axes[2].set_title("Market Capitalization Growth (INR Lakh Crores)")
axes[2].set_ylabel("Market Cap (Lakh Crores)")
axes[2].legend()

# Adjust layout and show plots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
plt.show()