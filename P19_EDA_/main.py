import pandas as pd
#from ydata_profiling import ProfileReport
from data_profiling import ProfileReport  # pip install fg-data-profiling

df = pd.read_csv("../P19_EDA_/iris.csv")

reporte = ProfileReport(
    df,
    title="Análisis exploratorio de datos",
    explorative=True
)

# Para Jupyter
#reporte.to_notebook_iframe()

#Para HTML
reporte.to_file("reporte_explorative_abalone.html")

