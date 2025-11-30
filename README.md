# 🥊 Optimization Battle Arena: AdaGrad vs. AdaDelta

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Math-green)](https://numpy.org/)

An interactive playground to visualize, compare, and understand the inner workings of Deep Learning optimization algorithms.

This project implements **AdaGrad** and **AdaDelta** entirely from scratch using **NumPy**, visualizing their convergence paths on a complex loss surface to demonstrate why adaptive learning rates are crucial.

![App Screenshot](assets/playground_screenshot.png)
*(Make sure your image is named `playground_screenshot.png` inside the `assets` folder, or update this link!)*

## 🧐 The Problem
In Deep Learning, standard Gradient Descent often struggles with complex terrains (saddle points, ravines).
* **AdaGrad** introduced adaptive learning rates but suffers from the **"Dying Learning Rate"** problem (the accumulator grows indefinitely, causing the step size to vanish).
* **AdaDelta** was proposed to solve this by limiting the window of accumulated past gradients and eliminating the need for a manual learning rate.

**The Goal:** create a visual proof-of-concept to verify if AdaDelta actually outperforms AdaGrad in these edge cases.

## 🛠️ Features
* **From-Scratch Implementation:** No `torch.optim` or `tf.keras`. The mathematical formulas for the optimizers are translated directly into NumPy vector operations.
* **Interactive Playground:** Built with **Streamlit**. Users can tweak:
    * Learning Rates (for AdaGrad).
    * Decay Rates / Rho (for AdaDelta).
    * Starting positions.
    * Iteration counts.
* **Real-time Visualization:** Uses **Matplotlib** contour plots to render the 2D loss landscape ($f(x,y) = x^2 + 20y^2$) and trace the optimization path.
* **Metric Comparison:** Automatically calculates the Euclidean distance to the global minimum to declare a winner.

## 🧮 The Math Behind the Code

### AdaGrad (The Challenger)
Accumulates the sum of squared gradients to adapt the learning rate.
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot g_t$$
*Weakness: As $G_t$ grows, the effective learning rate approaches 0.*

### AdaDelta (The Champion)
Uses Exponential Moving Averages (EMA) to limit the window of past gradients and corrects unit mismatches.
$$\Delta \theta_t = - \frac{\text{RMS}[\Delta \theta]_{t-1}}{\text{RMS}[g]_t} \cdot g_t$$
*Strength: No manual learning rate ($\eta$) required. Robust to scale.*

## 🚀 How to Run

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/monsif1999/adagrad-vs-adadelta.git](https://github.com/monsif1999/adagrad-vs-adadelta.git)
    cd adagrad-vs-adadelta
    ```

2.  **Install dependencies**
    ```bash
    pip install numpy matplotlib streamlit
    ```

3.  **Launch the App**
    ```bash
    streamlit run app.py
    ```

## 🧠 Key Learnings
Building this project highlighted several critical concepts in optimization theory:
1.  **Theory