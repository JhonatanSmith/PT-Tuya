import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -- Configuración de la página
st.set_page_config(
    page_title="Segmentación de Clientes & Descuentos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Título
st.title("💼 Segmentación de Clientes y Asignación de Descuentos")

# -- Carga de datos
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

data_path = "results/data/clustered_data.csv"
df = load_data(data_path)

# -- Si falta 'descuento_%', lo calculamos dinámicamente
if 'descuento_%' not in df.columns:
    mediana_1 = df[df['cluster']==1]['ratio_avance_compra'].median()
    def asignar_descuento(row):
        if row['cluster'] == 0:
            return 0
        if row['cluster'] == 2:
            return 25
        return 20 if row['ratio_avance_compra'] > mediana_1 else 5
    df['descuento_%'] = df.apply(asignar_descuento, axis=1)

# -- Sidebar: Resumen del Análisis
st.sidebar.header("🔍 Resumen del Análisis")
st.sidebar.markdown(
    """
    **Modelo:** K-Means (k=3)
    - **Silhouette:** 0.54
    - **Calinski-Harabasz:** 84.7
    - **Davies-Bouldin:** 0.60
    
    **Clusters & Descuentos:**
    - Cluster 0: 0%
    - Cluster 1: 5% o 20% según ratio
    - Cluster 2: 25%
    """
)

# -- Resumen de descuentos
st.subheader("📊 Resumen de Descuentos Asignados")
st.write(
    df['descuento_%']
      .value_counts()
      .rename_axis('Descuento (%)')
      .reset_index(name='Clientes')
)

# -- Páginas de la app
page = st.sidebar.radio("Selecciona página:", ["Análisis Descriptivo", "Segmentación"])

if page == "Análisis Descriptivo":
    st.header("📋 Análisis Descriptivo")
    st.markdown("Este panel muestra las visualizaciones exploratorias antes de modelar clusters.")
    # Listado de imágenes descriptivas en dos columnas
    pics = [
        "actividad_cliente.png",
        "box_monto_trans.png",
        "hit_valor.png",
        "tipo_cliente.png",
        "top_clientes_transacciones.png",
        "top_clientes_valor.png",
        "trasnacion.png"
    ]
    cols = st.columns(2)
    for idx, pic in enumerate(pics):
        col = cols[idx % 2]
        with col:
            caption = pic.replace('.png','').replace('_',' ')
            st.image(f"results/pics/{pic}", caption=caption, use_container_width=True)
    # Salir de la página descriptiva
    st.stop()

# -- Segmentación (página principal existente)
st.subheader("📈 Visualizaciones de Clusters")
col1, col2 = st.columns(2)
st.subheader("📈 Visualizaciones de Clusters")
col1, col2 = st.columns(2)

# Scatter plot + imagen adicional
with col1:
    st.markdown("**Scatter: Total Compra vs Total Avance**")
    fig1, ax1 = plt.subplots(figsize=(7,5))
    sns.scatterplot(
        data=df, x="total_valor_compra", y="total_valor_avance",
        hue="cluster", palette="tab10", s=80, alpha=0.7, ax=ax1
    )
    ax1.set_xlabel("Total valor compras ($)")
    ax1.set_ylabel("Total valor avances ($)")
    ax1.legend(title="Cluster")
    st.pyplot(fig1)
    # Imagen de evolución mensual de transacciones
    st.markdown("**Evolución de transacciones por mes**")
    st.image("results/pics/transcciones_mes.png", use_container_width=True)

# Boxplots separados
with col2:
    st.markdown("**Boxplot: Número de Transacciones por Cluster**")
    fig2, ax2 = plt.subplots(figsize=(7,5))
    sns.boxplot(x="cluster", y="num_transacciones", data=df, ax=ax2)
    ax2.set_title("# Transacciones")
    st.pyplot(fig2)
    
    st.markdown("**Boxplot: Recencia (días) por Cluster**")
    fig3, ax3 = plt.subplots(figsize=(7,5))
    sns.boxplot(x="cluster", y="dias_desde_ultima_tx", data=df, ax=ax3)
    ax3.set_title("Recencia (días)")
    st.pyplot(fig3)

# -- Botones para mostrar clientes por cluster
st.subheader("👥 Listado de Clientes por Cluster")
clusters = sorted(df['cluster'].unique())
btn_cols = st.columns(len(clusters))
for idx, cl in enumerate(clusters):
    if btn_cols[idx].button(f"Ver Cluster {cl}"):
        st.write(f"**Clientes en Cluster {cl}**")
        st.dataframe(df[df['cluster']==cl][['Id_cliente','cluster','descuento_%']])

# -- Estadísticas resumidas
st.subheader("🔍 Estadísticas por Cluster")
tabs = st.tabs([f"Cluster {i}" for i in clusters])
for i, tab in enumerate(tabs):
    with tab:
        subset = df[df['cluster']==i]
        st.write(f"**Cluster {i}: {len(subset)} clientes**")
        st.write(subset.describe().T[['mean','std','min','max']])

# -- Footer
st.markdown("---")
st.markdown("Fuente: `master_table.csv` procesada y clusterizada con K-Means.")
