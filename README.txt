========================================================================
TALLER: SIMULADOR INTERACTIVO Y EVALUADOR DE CONCENTRACIÓN DE MERCADO
Curso: Organización Industrial
========================================================================

DESCRIPCIÓN DEL PROYECTO:
Aplicación web interactiva desarrollada en Python mediante el framework 
Streamlit para el cálculo, análisis estocástico y evaluación pedagógica 
de cuatro indicadores normativos de concentración económica:
1. Ratio de Concentración (CRk).
2. Índice de Herfindahl-Hirschman (IHH).
3. Índice de Dominancia (ID).
4. Índice de Entropía (IE).

CARACTERÍSTICAS TÉCNICAS:
- Simulación estocástica de Monte Carlo (100 a 10.000 iteraciones) utilizando 
  la distribución Dirichlet para garantizar cuotas homogéneas que suman 1.0.
- Comparación empírica con gráficos de densidad (KDE Gaussiano) e histogramas 
  en Matplotlib, marcando el valor particular y su percentil exacto.
- Módulo evaluador interactivo con feedback pedagógico y umbrales teóricos.
- Alerta preventiva de latencia para simulaciones superiores a 3.000 iteraciones.

REQUISITOS DEL SISTEMA:
- Python 3.9 o superior.
- Librerías listadas en requirements.txt:
  * streamlit
  * numpy
  * matplotlib
  * scipy
  * pandas

INSTRUCCIONES DE EJECUCIÓN LOCAL:
1. Abrir la terminal en la carpeta raíz del proyecto.
2. Instalar las dependencias necesarias:
   pip install -r requirements.txt
3. Ejecutar la aplicación web con Streamlit:
   python -m streamlit run app.py
4. La aplicación se abrirá automáticamente en http://localhost:8501
========================================================================
