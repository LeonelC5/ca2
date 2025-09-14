# 🎓 Student GPA Predictor

## Problema
Desarrollar una herramienta predictiva que estime el **GPA final** de estudiantes universitarios de primer semestre, con el objetivo de **identificar tempranamente a quienes podrían necesitar apoyo académico**. La solución debe ser ética, motivacional y ofrecer recomendaciones prácticas para mejorar el desempeño.

## Tarea (Task)
- Predecir un **valor numérico continuo**: el GPA final de cada estudiante.  
- El enfoque principal será la **regresión**, utilizando técnicas lineales o no lineales según corresponda.  
- La solución debe ser escalable y permitir futuras mejoras, como incluir interacciones entre variables o transformar la regresión lineal en una más compleja si se detectan relaciones no lineales.

## Métrica (Metric)
- Evaluar la calidad del modelo usando métricas de regresión, como:  
  - **MSE (Mean Squared Error / Error Cuadrático Medio)**: informa sobre la magnitud promedio de los errores.  
  - **R² (Coeficiente de Determinación)**: indica qué proporción de la variabilidad del GPA es explicada por el modelo.  
- La métrica debe ser **informativa y práctica**, permitiendo comparar versiones del modelo y monitorear mejoras post-deployment.

## Experiencia del Usuario (Experience)
- La herramienta debe ser **motivacional y no desalentadora**:  
  - Evitar mensajes negativos que puedan generar frustración.  
  - Generar recomendaciones **específicas y accionables** para cada estudiante (por ejemplo, horas de estudio semanales, participación en tutorías, hábitos de aprendizaje).  
- Evitar **sesgos por género, raza o clase social**:  
  - El modelo no debe discriminar ni generar predicciones que dependan de estas variables.  
- Diseñar **dos vistas distintas**:  
  1. **Vista para estudiantes**: enfocada en motivación, progreso y mejora continua.  
  2. **Vista para coordinadores académicos**: enfocada en identificación de riesgo, priorización de intervenciones y recursos disponibles.  
- La interfaz debe ser **útil y accesible**, resolviendo el problema central de los stakeholders: estudiantes y coordinadores académicos.


---

## 📊 Dataset
[Student Performance Prediction - Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset)  

### Variables consideradas en el modelo:
| Variable | Tipo | Descripción |
|----------|------|------------|
| Age | Numérica | Edad (15–18 años) |
| StudyTimeWeekly | Numérica | Horas de estudio semanales (0–20) |
| Absences | Numérica | Número de ausencias (0–30) |
| Tutoring | Categórica | Tutoría (0 = No, 1 = Sí) |
| ParentalSupport | Categórica | Nivel de apoyo parental (0 = Ninguno, 1 = Bajo, 2 = Moderado, 3 = Alto, 4 = Muy alto) |
| Extracurricular | Categórica | Actividades extracurriculares (0 = No, 1 = Sí) |
| Sports | Categórica | Deportes (0 = No, 1 = Sí) |
| Music | Categórica | Música (0 = No, 1 = Sí) |
| Volunteering | Categórica | Voluntariado (0 = No, 1 = Sí) |


---

## 📌 Clasificación de GPA
- **A** : GPA ≥ 3.5  
- **B** : 3.0 ≤ GPA < 3.5  
- **C** : 2.5 ≤ GPA < 3.0  
- **D** : 2.0 ≤ GPA < 2.5  
- **F** : GPA < 2.0  

---
## 🎯 Selección de Variables

Durante la preparación de datos se tomaron decisiones clave respecto a qué variables incluir en el modelo y cuáles excluir, con el objetivo de balancear **precisión, interpretabilidad y equidad**:

- **Variables eliminadas:**
  - `StudentID`: identificador único, sin valor predictivo.
  - `GradeClass` y `GPA`: variables objetivo o derivadas, no deben usarse como predictoras.
  - `Gender` y `Ethnicity`: eliminadas para evitar sesgos de género o raciales, priorizando un uso ético del modelo.

- **Variables parcialmente consideradas:**
  - `ParentalEducation`: se utilizó en el entrenamiento para analizar su impacto estadístico en el GPA, pero **se omitió en la aplicación**.  
    👉 Razón: aunque aporta algo de valor predictivo, podría introducir sesgos socioeconómicos. Se decidió mantenerlo en el análisis académico, pero excluirlo en la experiencia del usuario final.

- **Variables utilizadas en la aplicación:**
  - `StudyTimeWeekly`, `Absences`, `Tutoring`, `ParentalSupport`, `Extracurricular`, `Sports`, `Music`, `Volunteering`, `Age`.

Este enfoque asegura que la herramienta final sea **motivacional, justa y centrada en hábitos y apoyo académico**, sin depender de características sensibles o externas al control del estudiante.


## ⚙️ Modelo
Se utilizó **Regresión Lineal** para predecir el GPA de los estudiantes, considerando únicamente variables relacionadas con estudio, tutoría, actividades extracurriculares y apoyo parental.  

**Flujo del modelo:**  

1. **División de datos**:  
   - 80% para entrenamiento y 20% para prueba.  

2. **Entrenamiento**:  
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
```

3. **Evaluación del modelo**

```python
from sklearn.metrics import mean_squared_error

# Predecir GPA en el conjunto de prueba
y_pred = model.predict(x_test)

# Calcular error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```

4. **Guardado del modelo entrenado**

```python
import joblib

print("\n=== Importancia de variables ===")
if best_model[0] == "Lineal":
    for var, peso in zip(x_train.columns, linear.coef_):
        print(f"{var}: {peso:.4f}")
elif best_model[0] == "Polinómica (deg=2)":
    # Obtener nombres de features polinómicas
    feature_names = poly2.named_steps['poly'].get_feature_names_out(x_train.columns)
    for var, peso in zip(feature_names, poly2.named_steps['lin'].coef_):
        print(f"{var}: {peso:.4f}")
elif best_model[0] in ["Tree", "RandomForest"]:
    model_to_use = tree if best_model[0]=="Tree" else forest
    for var, imp in zip(x_train.columns, model_to_use.feature_importances_):
        print(f"{var}: {imp:.4f}")
```

## Resultados

- **Modelo entrenado:** Regresión Lineal  
- **Evaluación:** RMSE = 0.196, R² = 0.953 (mejor modelo según R²)  
- **Comparación de modelos:**  
  - Lineal: RMSE=0.196, R²=0.953  
  - Polinómica (deg=2): RMSE=0.201, R²=0.951  
  - Árbol de decisión: RMSE=0.332, R²=0.866  
  - Random Forest: RMSE=0.239, R²=0.931 

## ⚙️ Justificación de Modelos Probados

Para la construcción de la herramienta se exploraron cuatro enfoques diferentes de regresión y predicción, seleccionados por su **equilibrio entre interpretabilidad, capacidad predictiva y adecuación al tamaño del dataset**:

1. **Regresión Lineal**  
   - Se probó por ser el modelo más sencillo y altamente interpretable.  
   - Permite identificar de manera directa el peso de cada variable en la predicción del GPA.  
   - Es un buen punto de partida y baseline para comparar otros modelos más complejos.

2. **Regresión Polinómica (grado 2)**  
   - Se utilizó para capturar relaciones no lineales entre las variables (por ejemplo, que más horas de estudio no siempre implican un aumento proporcional en el GPA).  
   - Es una extensión natural de la regresión lineal que mantiene cierto nivel de interpretabilidad.

3. **Árbol de Decisión**  
   - Se evaluó porque ofrece interpretabilidad visual y maneja de manera natural variables categóricas.  
   - Permite explorar reglas de decisión simples (ejemplo: "si ausencias > X, el GPA disminuye").  
   - Sin embargo, tiende a sobreajustar si no se controla la profundidad.

4. **Random Forest**  
   - Se incluyó como un modelo más robusto frente al sobreajuste, combinando múltiples árboles de decisión.  
   - Captura interacciones complejas entre variables y mejora la estabilidad del modelo.  
   - Aunque menos interpretable que la regresión lineal, es un referente en tareas de predicción con datasets pequeños-medianos.

Estos cuatro modelos se seleccionaron porque representan un espectro balanceado: **linealidad, no linealidad controlada, interpretabilidad, y robustez**, lo que permitió comparar enfoques y elegir el que ofreciera mejor desempeño y claridad en los resultados.

---
- **Variables más influyentes en la predicción del GPA:**  
1. **StudyTimeWeekly** – horas de estudio semanales **(+0.0291)**
2. **Tutoring** – participación en tutorías **(+0.2576)**
3. **ParentalSupport** – nivel de apoyo parental **(+0.1479)** 

## Otras variables que aportan al modelo
- **Extracurricular** – participación en actividades extracurriculares **(+0.1900)**
- **Sports** – práctica de deportes **(+0.1842)**
- **Music** – participación en música **(+0.1513)**
- **Absences** – número de faltas a clase **(-0.0995)**
- **Volunteering** – voluntariado **(-0.0049)**
- **Age** – edad del estudiante **(-0.0057)**
- **ParentalEducation_1** – nivel educativo padres (categoría 1) **(-0.0019)**
- **ParentalEducation_2** – nivel educativo padres (categoría 2) **(+0.0082)**
- **ParentalEducation_3** – nivel educativo padres (categoría 3) **(-0.0120)**
- **ParentalEducation_4** – nivel educativo padres (categoría 4) **(+0.0149)**


---

## Discusión

El desarrollo de la herramienta predictiva de GPA permitió analizar y modelar de manera efectiva el desempeño académico de estudiantes de primer semestre. A partir de los modelos evaluados (Regresión Lineal, Regresión Polinómica, Árbol de Decisión y Random Forest), la **Regresión Lineal** se identificó como la opción más adecuada, con un **R² de 0.953** y **RMSE de 0.196**, demostrando un ajuste muy preciso a los datos y manteniendo simplicidad e interpretabilidad frente a otros enfoques más complejos.

### Interpretación de resultados
Las variables con mayor impacto en la predicción del GPA fueron:

- **Tutoring**: La participación en tutorías mostró la mayor influencia positiva, indicando que el acompañamiento académico directo tiene un efecto significativo en el desempeño.  
- **StudyTimeWeekly**: Las horas de estudio semanales se correlacionan positivamente con un mejor GPA, confirmando la importancia de la dedicación al estudio.  
- **ParentalSupport**: El apoyo familiar también mostró un impacto relevante, aunque menor que la participación activa en actividades académicas.  

Otras variables, como **Extracurricular, Sports, Music y Volunteering**, también aportaron al modelo, reflejando cómo la participación en actividades complementarias contribuye al bienestar y la motivación del estudiante.  

Variables demográficas como **Gender, Ethnicity y ParentalEducation** fueron analizadas en el entrenamiento, pero **no se consideraron en la aplicación** para evitar sesgos y garantizar una experiencia justa para el usuario.

### Ética y diseño de la app
Un aspecto crítico del proyecto fue garantizar que la herramienta **sea motivacional y libre de sesgos**. Para lograrlo:

- La interfaz de usuario **no solicita información sensible** (género, etnia o educación de los padres), eliminando cualquier posibilidad de sesgo en la interacción.  
- La retroalimentación es constructiva, diferenciando entre estudiantes que necesitan apoyo y aquellos con buen desempeño, **enfocándose en acciones concretas y motivacionales**.  
- La app ofrece dos vistas:  
  - **Estudiante:** Consejos personalizados y motivacionales.  
  - **Coordinador:** Identificación de estudiantes en riesgo y recomendaciones de intervención.

### Limitaciones
- El modelo se entrenó con datos de estudiantes de primer semestre, por lo que su generalización a otros ciclos podría ser limitada.  
- La exclusión de variables demográficas, si bien ética, puede haber eliminado cierta información estadísticamente relevante; sin embargo, este compromiso fue necesario para priorizar la equidad.  
- La herramienta depende de la correcta entrada de datos por parte del usuario; errores en el registro de horas de estudio o ausencias pueden afectar la predicción.

### Conclusión
La herramienta demuestra que es posible crear un sistema predictivo de desempeño académico **preciso, motivacional y ético**. La selección de variables centradas en hábitos y apoyo académico permite generar recomendaciones útiles sin introducir sesgos, cumpliendo con el objetivo de identificar estudiantes que requieren intervención temprana y fomentar hábitos positivos desde el inicio de su vida universitaria.
