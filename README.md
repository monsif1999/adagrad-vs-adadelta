# AdaGrad vs. AdaDelta : Combat d'Optimisation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)

Visualisation interactive comparant les optimiseurs AdaGrad et AdaDelta sur une surface de perte 2D.

![App Screenshot](assets/playground.png)

## Pourquoi ce projet

AdaGrad adapte les taux d'apprentissage mais l'accumulateur croît indéfiniment, ce qui fait que les pas rétrécissent jusqu'à zéro. AdaDelta corrige ça avec des moyennes mobiles exponentielles. Je voulais voir la différence visuellement.

## Ce qu'il fait

- Implémente les deux optimiseurs from scratch en NumPy (sans PyTorch/TensorFlow)
- Montre les chemins de convergence sur un graphique de contour ($f(x,y) = x^2 + 20y^2$)
- Permet d'ajuster les taux d'apprentissage, decay, points de départ, itérations
- Compare les distances finales au minimum

## Lancer l'app
```bash
git clone https://github.com/monsif1999/adagrad-vs-adadelta.git
cd adagrad-vs-adadelta
pip install numpy matplotlib streamlit
streamlit run app.py
```

## Les maths

**AdaGrad :**
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot g_t$$

Problème : $G_t$ continue de croître, donc les mises à jour meurent.

**AdaDelta :**
$$\Delta \theta_t = - \frac{\text{RMS}[\Delta \theta]_{t-1}}{\text{RMS}[g]_t} \cdot g_t$$

Solution : Utilise des moyennes mobiles au lieu d'une accumulation infinie. Pas besoin de taux d'apprentissage manuel.

## Ce que j'ai appris

- Pourquoi les taux d'apprentissage adaptatifs comptent sur des surfaces bizarres
- AdaGrad marche jusqu'à ce qu'il ne marche plus (regardez le taux d'apprentissage s'effondrer)
- L'astuce RMS d'AdaDelta fonctionne vraiment en pratique