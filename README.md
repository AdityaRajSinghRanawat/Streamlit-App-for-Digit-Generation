
# ✍️ Handwritten Digit Generator (0–9)

A Streamlit-based web app that generates **5 unique handwritten images** of any digit (0–9), trained from scratch using the **MNIST dataset** and a **simple conditional generator model** (PyTorch).

🔗 [Live Demo](https://your-app-link.streamlit.app)  
📄 Part of the METI Internship AI Task (Problem 3)

---

## 🚀 Features

- 🔢 User selects a digit (0–9)
- 🧠 App generates 5 diverse handwritten images of that digit
- 💻 Model trained from scratch (no pre-trained weights used)
- 🌐 Fully interactive Streamlit web interface

---

## 🧠 Model Overview

The generator model is a simple fully-connected neural network that:
- Accepts a **100-dim noise vector**
- Takes the **digit label** as a conditional input
- Outputs a **28×28 grayscale image**

Trained using Mean Squared Error (MSE) loss to approximate MNIST digits.

---

## 🗂️ Project Structure

