import pandas as pd
import numpy as np

# Limpieza de datos - Funciones fecha
def filtrar_fechas_validas(df, col_fecha):
    """
    Filtra un DataFrame eliminando registros con fechas no válidas.
    Se espera que las fechas estén en formato numérico 'YYYYMMDD' (8 dígitos).
    
    Parámetros:
    - df: DataFrame original
    - col_fecha: nombre de la columna que contiene las fechas
    
    Retorna:
    - df_limpio: DataFrame con solo fechas válidas
    - df_invalidos: DataFrame con fechas no válidas
    
    Ejemplo de uso:
        
        detalle_cliente_valido, detalle_cliente_invalidos = filtrar_fechas_validas(detalle_cliente, "fecha_efectiva")

    """
    
    # Eliminar filas con fecha nula
    df = df.dropna(subset=[col_fecha]).copy()
    
    # Convertir a string de 8 dígitos (eliminando ".0" si viene como float)
    df[col_fecha] = df[col_fecha].astype(float).astype("Int64").astype(str)
    
    # Patrón regex para fechas tipo 'YYYYMMDD'
    pattern = r"^\d{8}$"
    
    # Separar válidos e inválidos
    df_validos = df[df[col_fecha].str.match(pattern, na=False)]
    df_invalidos = df[~df[col_fecha].str.match(pattern, na=False)]
    
    # Mostrar log de inválidos
    if not df_invalidos.empty:
        print(f"Se encontraron {len(df_invalidos)} registros con formato de fecha no válido:")
        print(df_invalidos[[col_fecha]])
    else:
        print("✅ Todas las fechas tienen formato válido.")
    
    return df_validos, df_invalidos


# Limpieza de datos - Valores numéricos
def limpiar_columna_numerica(df, col):
    """
    Limpia una columna del DataFrame convirtiendo a float los valores válidos y
    reemplazando los no convertibles con NaN. Luego elimina los registros con NaN en esa columna.

    Parámetros:
    - df: DataFrame original
    - col: nombre de la columna a limpiar (string)

    Retorna:
    - df_limpio: DataFrame con la columna limpia y sin valores no numéricos
    - n_invalidos: número de valores mal imputados que fueron eliminados
    
    Ejemplo de uso:
    detalle_tx_limpio, errores = limpiar_columna_numerica(detalle_tx, "valor")

    """

    # Función auxiliar para validar si se puede convertir a float
    def es_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    df = df.copy()  # Para no modificar el DataFrame original

    # Convertir a float o NaN
    df[col] = df[col].apply(lambda x: float(x) if es_float(x) else np.nan)

    # Contar cuántos valores inválidos fueron detectados
    n_invalidos = df[col].isna().sum()

    if n_invalidos > 0:
        print(f"⚠️ Se encontraron {n_invalidos} valores mal imputados en la columna '{col}' y fueron eliminados.")
    else:
        print(f"✅ Todos los valores en la columna '{col}' eran válidos.")

    # Eliminar registros con NaN en la columna
    df = df.dropna(subset=[col])

    return df, n_invalidos
