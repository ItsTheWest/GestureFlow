<div align="center">

# 🖐️ GestureFlow

**Real-time hand gesture recognition pipeline for Linux desktop control**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow\&logoColor=white)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green?logo=google\&logoColor=white)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

> A full end-to-end pipeline that captures hand landmarks via webcam, trains an LSTM neural network on custom gesture sequences, and translates recognized gestures into real OS actions — cursor movement, click, and workspace switching.

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Workflow](#-workflow)
- [Pipeline Steps](#-pipeline-steps)
- [Installation](#-installation)
- [Running the Dashboard](#-running-the-dashboard)
- [Configuration](#-configuration)
- [Tech Stack](#-tech-stack)

---

## 🧠 Overview

GestureFlow is a structured learning project that walks through the full Machine Learning lifecycle applied to computer vision:

| Phase | Description |
|---|---|
| **Exploration** | Raw camera capture and landmark visualization (Steps 1–3) |
| **Rule-based recognition** | Static gesture detection without neural networks (Step 4) |
| **Data pipeline** | Automated sequence collection into `.npy` datasets (Step 5) |
| **Training** | LSTM model training with temporal sequence data (Step 6) |
| **Inference** | Real-time gesture classification via the trained model (Step 7) |
| **System control** | OS-level actions driven by recognized gestures (Step 8) |

The final result is a unified **CustomTkinter dashboard** (`main.py`) that integrates steps 4–8 into a single dark-themed GUI with an embedded live camera viewport.
