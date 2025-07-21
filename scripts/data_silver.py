import pandas as pd
import numpy as np
from utils import filtrar_fechas_validas, limpiar_columna_numerica

# Lectura de datos
detalle_cliente = pd.read_excel("../data/bronze/Prueba proceso de selección Analista de Datos - bases.xlsx", 
                                sheet_name="Detalle_cliente")

detalle_tx = pd.read_excel("../data/bronze/Prueba proceso de selección Analista de Datos - bases.xlsx",
                           sheet_name = "Detalle_tx" )

# Correcion fecha detalle cliente
detalle_cliente, dc_invalido = filtrar_fechas_validas(detalle_cliente, "fecha_efectiva")
detalle_cliente["fecha_efectiva"] = pd.to_datetime(detalle_cliente["fecha_efectiva"], format="%Y%m%d", errors="coerce")

# Correcion valor detalle_tx
detalle_tx, errores = limpiar_columna_numerica(detalle_tx, "valor")

# Merge de DataFrames
df = detalle_cliente.merge(detalle_tx, on="Id_tx", how="left")

# Guardando resultados
df.to_csv("../data/silver/master_table.csv", index=False)
