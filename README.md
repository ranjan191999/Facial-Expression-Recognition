# Facial Expression Recognition & Emotion-Based Review Analysis

This project combines **Facial Emotion Recognition** with a **Review Generation and Retrieval System** built on FAISS and NLP.  
It was developed as part of a Junior AI Engineer assignment to demonstrate both **deep learning** and **applied AI integration** skills.

---

## Project Overview

The system is designed in two stages:

### **Stage 1 – Facial Expression Recognition**
A ResNet-18 model is fine-tuned to classify human facial emotions into:
> angry, disgust, fear, happy, sad, surprise, neutral

### **Stage 2 – Review Generation + RAG**
Using the predicted emotion (for example, “happy”), the system:
1. Generates synthetic reviews with natural-language generation models  
2. Embeds them using `sentence-transformers`  
3. Stores and retrieves them efficiently with **FAISS**  
4. Answers queries or summarizes what users of that emotion are “saying”  

---

## Installation

### Clone the repository
```bash
git clone https://github.com/ranjan191999/Facial-Expression-Recognition.git
cd Facial-Expression-Recognition