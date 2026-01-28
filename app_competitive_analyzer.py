"""
Competitive Analyzer - Streamlit App
=====================================
Interfaz gráfica profesional para análisis competitivo multi-dimensional

Autor: Analytics Team
Versión: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from competitive_analyzer import CompetitiveAnalyzer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Competitive Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    h1 {
        color: #1E88E5;
        padding-bottom: 10px;
        border-bottom: 3px solid #1E88E5;
    }
    h2 {
        color: #43A047;
        margin-top: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin-bottom: 1rem;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin-bottom: 1rem;
    }
    .danger-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def initialize_session_state():
    """Inicializa el estado de la sesión"""
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

def save_uploaded_file(uploaded_file):
    """Guarda el archivo subido temporalmente"""
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def create_price_evolution_chart(df, date_col, price_vars):
    """Crea gráfico de evolución de precios"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for i, (brand, col) in enumerate(price_vars.items()):
        if col in df.columns:
            # Usar el nombre de la columna directamente
            label = col if brand != 'client' else f"{col} (Cliente)"
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[col],
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=3),
                marker=dict(size=8)
            ))
    
    fig.update_layout(
        title='Evolución de Precios - Análisis Competitivo',
        xaxis_title='Período',
        yaxis_title='Precio ($)',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_price_index_chart(df, date_col):
    """Crea gráfico de índice de precios relativos"""
    fig = go.Figure()
    
    # Buscar columnas de índice de precio
    index_cols = [col for col in df.columns if 'PriceIndex_' in col]
    
    colors = px.colors.qualitative.Set1
    
    for i, col in enumerate(index_cols):
        # Extraer nombre del competidor
        competitor = col.replace('PriceIndex_', '')
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[col],
            mode='lines+markers',
            name=f'vs {competitor}',
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6)
        ))
    
    # Línea de paridad (100 = precios iguales)
    fig.add_hline(y=100, line_dash="dash", line_color="gray", 
                  annotation_text="Paridad (100)", annotation_position="right")
    
    fig.update_layout(
        title='Índice de Precio Relativo - Cliente vs Competencia',
        xaxis_title='Período',
        yaxis_title='Índice (100 = Paridad)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_units_market_share_chart(df, date_col):
    """Crea gráfico de market share en unidades"""
    fig = go.Figure()
    
    # Buscar columnas de market share
    ms_cols = [col for col in df.columns if 'MS_Units_' in col]
    
    colors = px.colors.qualitative.Set2
    
    for i, col in enumerate(ms_cols):
        brand = col.replace('MS_Units_', '')
        # Usar el nombre real de la columna
        label = f"{brand} (Cliente)" if brand == 'client' else brand
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[col],
            mode='lines+markers',
            name=label,
            line=dict(color=color, width=3),
            marker=dict(size=8),
            stackgroup='one',
            groupnorm='percent'
        ))
    
    fig.update_layout(
        title='Market Share en Unidades (%)',
        xaxis_title='Período',
        yaxis_title='Market Share (%)',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_value_comparison_chart(df, date_col, value_vars):
    """Crea gráfico comparativo de ventas en valor"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Ventas en Valor Absoluto', 'Market Share en Valor (%)'),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )
    
    colors = px.colors.qualitative.Set2
    
    # Gráfico 1: Valores absolutos
    for i, (brand, col) in enumerate(value_vars.items()):
        if col in df.columns:
            label = f"{col} (Cliente)" if brand == 'client' else col
            color = colors[i % len(colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=df[date_col],
                    y=df[col],
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=3),
                    legendgroup=label
                ),
                row=1, col=1
            )
    
    # Gráfico 2: Market share
    ms_cols = [col for col in df.columns if 'MS_Value_' in col]
    
    for i, col in enumerate(ms_cols):
        brand = col.replace('MS_Value_', '')
        label = f"{brand} (Cliente)" if brand == 'client' else brand
        color = colors[i % len(colors)]
        
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[col],
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=3),
                marker=dict(size=6),
                legendgroup=label,
                showlegend=False
            ),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Período", row=2, col=1)
    fig.update_yaxes(title_text="Ventas ($)", row=1, col=1)
    fig.update_yaxes(title_text="Market Share (%)", row=2, col=1)
    
    fig.update_layout(
        height=800,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_distribution_analysis_chart(df, date_col, dist_vars):
    """Crea gráfico de análisis de distribución"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Puntos de Venta (PDV)', 'Share de Distribución (%)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = px.colors.qualitative.Set2
    
    # Gráfico 1: PDV absolutos
    for i, (brand, col) in enumerate(dist_vars.items()):
        if col in df.columns:
            label = f"{col} (Cliente)" if brand == 'client' else col
            color = colors[i % len(colors)]
            
            fig.add_trace(
                go.Bar(
                    x=df[date_col],
                    y=df[col],
                    name=label,
                    marker_color=color,
                    legendgroup=label
                ),
                row=1, col=1
            )
    
    # Gráfico 2: Share de distribución
    sd_cols = [col for col in df.columns if 'SD_' in col]
    
    for i, col in enumerate(sd_cols):
        brand = col.replace('SD_', '')
        label = f"{brand} (Cliente)" if brand == 'client' else brand
        color = colors[i % len(colors)]
        
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[col],
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=3),
                marker=dict(size=8),
                legendgroup=label,
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Período", row=1, col=1)
    fig.update_xaxes(title_text="Período", row=1, col=2)
    fig.update_yaxes(title_text="PDV", row=1, col=1)
    fig.update_yaxes(title_text="Share (%)", row=1, col=2)
    
    fig.update_layout(
        height=500,
        template='plotly_white',
        barmode='group'
    )
    
    return fig

def create_google_trends_chart(df, date_col, gt_vars):
    """Crea gráfico de Google Trends"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Interés de Búsqueda (Índice)', 'Share of Search (%)'),
        vertical_spacing=0.15
    )
    
    colors = px.colors.qualitative.Set2
    
    # Gráfico 1: Interés absoluto
    for i, (brand, col) in enumerate(gt_vars.items()):
        if col in df.columns:
            label = f"{col} (Cliente)" if brand == 'client' else col
            color = colors[i % len(colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=df[date_col],
                    y=df[col],
                    mode='lines+markers',
                    name=label,
                    line=dict(color=color, width=3),
                    marker=dict(size=8),
                    legendgroup=label
                ),
                row=1, col=1
            )
    
    # Gráfico 2: Share of Search
    sos_cols = [col for col in df.columns if 'SoS_' in col]
    
    for i, col in enumerate(sos_cols):
        brand = col.replace('SoS_', '')
        label = f"{brand} (Cliente)" if brand == 'client' else brand
        color = colors[i % len(colors)]
        
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[col],
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=3),
                marker=dict(size=6),
                legendgroup=label,
                showlegend=False
            ),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Período", row=2, col=1)
    fig.update_yaxes(title_text="Índice", row=1, col=1)
    fig.update_yaxes(title_text="Share (%)", row=2, col=1)
    
    fig.update_layout(
        height=800,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_competitive_radar(metrics):
    """Crea gráfico radar de posicionamiento competitivo"""
    categories = []
    client_values = []
    
    # Precio (invertido - menor es mejor)
    if 'precios' in metrics and 'positioning' in metrics['precios']:
        categories.append('Precio<br>Competitivo')
        premium_idx = metrics['precios']['positioning']['premium_index']
        # Convertir a escala 0-100 donde 100 es mejor
        price_score = max(0, 100 - abs(premium_idx))
        client_values.append(price_score)
    
    # Unidades - crecimiento
    if 'unidades' in metrics and 'ms_client' in metrics['unidades']:
        categories.append('Market Share<br>Unidades')
        ms = metrics['unidades']['ms_client']['actual']
        client_values.append(ms)
    
    # Valor - market share
    if 'valor' in metrics and 'ms_client' in metrics['valor']:
        categories.append('Market Share<br>Valor')
        ms = metrics['valor']['ms_client']['actual']
        client_values.append(ms)
    
    # Distribución - share
    if 'distribucion' in metrics and 'share_client' in metrics['distribucion']:
        categories.append('Share de<br>Distribución')
        sd = metrics['distribucion']['share_client']['actual']
        client_values.append(sd)
    
    # Google Trends - share of search
    if 'google_trends' in metrics and 'sos_client' in metrics['google_trends']:
        categories.append('Share of<br>Search')
        sos = metrics['google_trends']['sos_client']['actual']
        client_values.append(sos)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=client_values,
        theta=categories,
        fill='toself',
        name='Cliente',
        line=dict(color='#1E88E5', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title='Radar Competitivo Multi-Dimensional',
        height=600
    )
    
    return fig

def create_correlation_heatmap(df, vars_dict):
    """Crea heatmap de correlaciones"""
    # Seleccionar variables relevantes
    all_vars = []
    for var_type, vars_group in vars_dict.items():
        all_vars.extend([v for v in vars_group.values() if v in df.columns])
    
    if len(all_vars) < 2:
        return None
    
    # Calcular matriz de correlación
    corr_matrix = df[all_vars].corr()
    
    # Crear labels más cortos
    labels = [col.replace('_LIST', '').replace('_Colgate', '_CG').replace('_OralB', '_OB') 
              for col in corr_matrix.columns]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlación")
    ))
    
    fig.update_layout(
        title='Matriz de Correlaciones - Variables Competitivas',
        height=600,
        xaxis={'side': 'bottom'}
    )
    
    return fig

# ============================================================================
# APLICACIÓN PRINCIPAL
# ============================================================================

initialize_session_state()

# ============================================================================
# SIDEBAR - CONFIGURACIÓN
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/combo-chart.png", width=100)
    st.title("⚙️ Configuración")
    
    # Upload de archivo
    st.markdown("### 📁 Cargar Datos")
    uploaded_file = st.file_uploader(
        "Selecciona tu archivo Excel",
        type=['xlsx', 'xls'],
        help="Archivo con datos mensuales de mercado"
    )
    
    if uploaded_file is not None:
        if st.session_state.uploaded_file != uploaded_file.name:
            st.session_state.uploaded_file = uploaded_file.name
            st.session_state.analysis_complete = False
            
            # Guardar archivo
            file_path = save_uploaded_file(uploaded_file)
            
            # Crear analizador
            st.session_state.analyzer = CompetitiveAnalyzer(
                filepath=file_path,
                date_column='Date'
            )
            
            st.success("✅ Archivo cargado correctamente")
    
    # Configuración de marcas
    if st.session_state.analyzer is not None:
        st.markdown("---")
        
        # Selector manual de variables
        st.markdown("### 🔧 Selección de Variables")
        
        if st.session_state.analyzer is not None:
            df = st.session_state.analyzer.df
            all_columns = [col for col in df.columns if col != 'Date']
            
            with st.expander("💰 Variables de PRECIO", expanded=False):
                st.markdown("**Cliente:**")
                precio_client = st.selectbox(
                    "Precio Cliente",
                    options=['Auto'] + all_columns,
                    key='precio_client'
                )
                
                st.markdown("**Competidores:**")
                precio_comp = st.multiselect(
                    "Precios Competencia",
                    options=all_columns,
                    key='precio_comp'
                )
            
            with st.expander("📦 Variables de UNIDADES", expanded=False):
                st.markdown("**Cliente:**")
                unid_client = st.selectbox(
                    "Unidades Cliente",
                    options=['Auto'] + all_columns,
                    key='unid_client'
                )
                
                st.markdown("**Competidores:**")
                unid_comp = st.multiselect(
                    "Unidades Competencia",
                    options=all_columns,
                    key='unid_comp'
                )
            
            with st.expander("💵 Variables de VALOR", expanded=False):
                st.markdown("**Cliente:**")
                valor_client = st.selectbox(
                    "Valor/Ventas Cliente",
                    options=['Auto'] + all_columns,
                    key='valor_client'
                )
                
                st.markdown("**Competidores:**")
                valor_comp = st.multiselect(
                    "Valor/Ventas Competencia",
                    options=all_columns,
                    key='valor_comp'
                )
            
            with st.expander("🏪 Variables de DISTRIBUCIÓN", expanded=False):
                st.markdown("**Cliente:**")
                dist_client = st.selectbox(
                    "Distribución Cliente",
                    options=['Auto'] + all_columns,
                    key='dist_client'
                )
                
                st.markdown("**Competidores:**")
                dist_comp = st.multiselect(
                    "Distribución Competencia",
                    options=all_columns,
                    key='dist_comp'
                )
            
            with st.expander("🔍 Variables de GOOGLE TRENDS", expanded=False):
                st.markdown("**Cliente:**")
                gt_client = st.selectbox(
                    "Google Trends Cliente",
                    options=['Auto'] + all_columns,
                    key='gt_client'
                )
                
                st.markdown("**Competidores:**")
                gt_comp = st.multiselect(
                    "Google Trends Competencia",
                    options=all_columns,
                    key='gt_comp'
                )
        
        # Botón de análisis
        st.markdown("---")
        if st.button("🚀 Ejecutar Análisis Completo", type="primary", width='stretch'):
            with st.spinner("Analizando datos..."):
                # Limpiar variables existentes
                st.session_state.analyzer.price_vars = {}
                st.session_state.analyzer.units_vars = {}
                st.session_state.analyzer.value_vars = {}
                st.session_state.analyzer.dist_vars = {}
                st.session_state.analyzer.gt_vars = {}
                
                # Aplicar variables manuales seleccionadas
                if st.session_state.get('precio_client') and st.session_state.precio_client != 'Auto':
                    st.session_state.analyzer.price_vars['client'] = st.session_state.precio_client
                if st.session_state.get('precio_comp'):
                    for col in st.session_state.precio_comp:
                        # Usar el nombre de la columna como clave
                        st.session_state.analyzer.price_vars[col] = col
                
                if st.session_state.get('unid_client') and st.session_state.unid_client != 'Auto':
                    st.session_state.analyzer.units_vars['client'] = st.session_state.unid_client
                if st.session_state.get('unid_comp'):
                    for col in st.session_state.unid_comp:
                        st.session_state.analyzer.units_vars[col] = col
                
                if st.session_state.get('valor_client') and st.session_state.valor_client != 'Auto':
                    st.session_state.analyzer.value_vars['client'] = st.session_state.valor_client
                if st.session_state.get('valor_comp'):
                    for col in st.session_state.valor_comp:
                        st.session_state.analyzer.value_vars[col] = col
                
                if st.session_state.get('dist_client') and st.session_state.dist_client != 'Auto':
                    st.session_state.analyzer.dist_vars['client'] = st.session_state.dist_client
                if st.session_state.get('dist_comp'):
                    for col in st.session_state.dist_comp:
                        st.session_state.analyzer.dist_vars[col] = col
                
                if st.session_state.get('gt_client') and st.session_state.gt_client != 'Auto':
                    st.session_state.analyzer.gt_vars['client'] = st.session_state.gt_client
                if st.session_state.get('gt_comp'):
                    for col in st.session_state.gt_comp:
                        st.session_state.analyzer.gt_vars[col] = col
                
                # Ejecutar análisis
                st.session_state.analyzer.run_full_analysis()
                st.session_state.analysis_complete = True
                
                st.success("✅ Análisis completado!")
                st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

if st.session_state.analyzer is not None and st.session_state.analysis_complete:
    
    analyzer = st.session_state.analyzer
    df = analyzer.df
    metrics = analyzer.metrics
    
    # Header
    st.title("🎯 Competitive Analyzer")
    st.markdown("**Análisis Competitivo Multi-Dimensional**")
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview",
        "💰 Precios",
        "📦 Unidades",
        "💵 Valor",
        "🏪 Distribución",
        "🔍 Google Trends",
        "🎯 Dashboard Integrado"
    ])
    
    # ========================================================================
    # TAB 1: OVERVIEW
    # ========================================================================
    
    with tab1:
        st.header("📊 Resumen Ejecutivo")
        
        # KPIs principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'precios' in metrics and 'client' in metrics['precios']:
                precio_actual = metrics['precios']['client']['actual']
                precio_cambio = metrics['precios']['client']['cambio_pct']
                st.metric(
                    "Precio Actual",
                    f"${precio_actual:,.0f}",
                    f"{precio_cambio:+.1f}%"
                )
        
        with col2:
            if 'unidades' in metrics and 'ms_client' in metrics['unidades']:
                ms_units = metrics['unidades']['ms_client']['actual']
                ms_cambio = metrics['unidades']['ms_client']['cambio_pp']
                st.metric(
                    "MS Unidades",
                    f"{ms_units:.1f}%",
                    f"{ms_cambio:+.1f} pp"
                )
        
        with col3:
            if 'valor' in metrics and 'ms_client' in metrics['valor']:
                ms_value = metrics['valor']['ms_client']['actual']
                ms_cambio = metrics['valor']['ms_client']['cambio_pp']
                st.metric(
                    "MS Valor",
                    f"{ms_value:.1f}%",
                    f"{ms_cambio:+.1f} pp"
                )
        
        with col4:
            if 'distribucion' in metrics and 'client' in metrics['distribucion']:
                dist_actual = metrics['distribucion']['client']['actual']
                dist_cambio = metrics['distribucion']['client']['cambio_pct']
                st.metric(
                    "Distribución",
                    f"{dist_actual:,.0f} PDV",
                    f"{dist_cambio:+.1f}%"
                )
        
        st.markdown("---")
        
        # Radar competitivo
        col1, col2 = st.columns([2, 1])
        
        with col1:
            radar_fig = create_competitive_radar(metrics)
            st.plotly_chart(radar_fig, width='stretch')
        
        with col2:
            st.markdown("### 🎯 Score Competitivo")
            
            # Calcular score
            favorable = 0
            total = 0
            
            checks = [
                ('precios', 'positioning', lambda m: m['category'] != 'premium'),
                ('unidades', 'tendencia', lambda m: m['direction'] == 'creciente'),
                ('valor', 'ms_client', lambda m: m['cambio_pp'] > 0),
                ('distribucion', 'fair_share', lambda m: m['status'] == 'over'),
                ('google_trends', 'momentum', lambda m: m['valor'] > 0)
            ]
            
            for metric_key, sub_key, check_func in checks:
                if metric_key in metrics and sub_key in metrics[metric_key]:
                    total += 1
                    if check_func(metrics[metric_key][sub_key]):
                        favorable += 1
            
            if total > 0:
                score = (favorable / total * 100)
                
                st.markdown(f"<div class='metric-card'>"
                          f"<h1>{score:.0f}%</h1>"
                          f"<p>{favorable}/{total} métricas favorables</p>"
                          f"</div>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if score >= 70:
                    st.markdown('<div class="success-box">✅ <strong>Posición competitiva FUERTE</strong></div>', 
                              unsafe_allow_html=True)
                elif score >= 50:
                    st.markdown('<div class="info-box">ℹ️ <strong>Posición competitiva MODERADA</strong></div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">⚠️ <strong>Posición competitiva DÉBIL</strong><br>Requiere plan de acción</div>', 
                              unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tabla de métricas clave
        st.markdown("### 📋 Métricas Clave por Dimensión")
        
        summary_data = []
        
        dimensions = [
            ('Precios', 'precios', 'client', 'actual', '$'),
            ('Unidades', 'unidades', 'ms_client', 'actual', '%'),
            ('Valor', 'valor', 'ms_client', 'actual', '%'),
            ('Distribución', 'distribucion', 'client', 'actual', 'PDV'),
            ('Google Trends', 'google_trends', 'client', 'actual', 'Índice')
        ]
        
        for dim_name, metric_key, sub_key, value_key, unit in dimensions:
            if metric_key in metrics and sub_key in metrics[metric_key]:
                value = metrics[metric_key][sub_key][value_key]
                cambio = metrics[metric_key][sub_key].get('cambio_pct', 
                         metrics[metric_key][sub_key].get('cambio_pp', 0))
                
                summary_data.append({
                    'Dimensión': dim_name,
                    'Valor Actual': f"{value:,.1f} {unit}",
                    'Cambio': f"{cambio:+.1f}%",
                    'Status': '✅' if cambio > 0 else '⚠️' if cambio < -5 else '➡️'
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, width='stretch', hide_index=True)
    
    # ========================================================================
    # TAB 2: PRECIOS
    # ========================================================================
    
    with tab2:
        st.header("💰 Análisis de Precios")
        
        if 'precios' in metrics:
            # Gráfico de evolución
            if analyzer.price_vars:
                fig_price_evol = create_price_evolution_chart(
                    df, analyzer.date_column, analyzer.price_vars
                )
                st.plotly_chart(fig_price_evol, width='stretch')
            
            # Índice de precio relativo
            st.markdown("### 📊 Índice de Precio Relativo")
            
            index_cols = [col for col in df.columns if 'PriceIndex_' in col]
            if index_cols:
                fig_price_index = create_price_index_chart(df, analyzer.date_column)
                st.plotly_chart(fig_price_index, width='stretch')
            
            # Métricas detalladas
            st.markdown("### 📋 Métricas de Precio")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Posicionamiento
                if 'positioning' in metrics['precios']:
                    pos = metrics['precios']['positioning']
                    st.markdown("#### 💎 Posicionamiento")
                    st.markdown(f"**Premium Index:** {pos['premium_index']:+.1f}%")
                    
                    if pos['category'] == 'premium':
                        st.markdown('<div class="info-box">🔵 Posicionamiento PREMIUM</div>', 
                                  unsafe_allow_html=True)
                    elif pos['category'] == 'discount':
                        st.markdown('<div class="success-box">🟢 Posicionamiento DISCOUNT</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box">⚪ Posicionamiento AT PAR</div>', 
                                  unsafe_allow_html=True)
            
            with col2:
                # Elasticidad
                if 'elasticity' in metrics['precios']:
                    elast = metrics['precios']['elasticity']
                    st.markdown("#### 🔗 Elasticidad Precio-Unidades")
                    st.markdown(f"**Correlación:** {elast['correlation']:.3f}")
                    
                    if elast['significant']:
                        if elast['correlation'] < -0.3:
                            st.markdown('<div class="warning-box">⚠️ Elasticidad NEGATIVA significativa</div>', 
                                      unsafe_allow_html=True)
                        elif elast['correlation'] > 0.3:
                            st.markdown('<div class="info-box">ℹ️ Efecto PREMIUM (correlación positiva)</div>', 
                                      unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box">ℹ️ Sin relación significativa</div>', 
                                  unsafe_allow_html=True)
            
            # Tabla comparativa
            st.markdown("### 📊 Comparativa de Precios")
            
            price_data = []
            for brand, stats in metrics['precios'].items():
                if isinstance(stats, dict) and all(k in stats for k in ['promedio', 'actual', 'cambio_pct', 'min', 'max', 'cv']):
                    # Usar el nombre real de la columna
                    label = f"{brand} (Cliente)" if brand == 'client' else brand
                    price_data.append({
                        'Marca': label,
                        'Precio Promedio': f"${stats['promedio']:,.0f}",
                        'Precio Actual': f"${stats['actual']:,.0f}",
                        'Cambio': f"{stats['cambio_pct']:+.1f}%",
                        'Min': f"${stats['min']:,.0f}",
                        'Max': f"${stats['max']:,.0f}",
                        'CV': f"{stats['cv']:.1f}%"
                    })
            
            if price_data:
                price_df = pd.DataFrame(price_data)
                st.dataframe(price_df, width='stretch', hide_index=True)
    
    # ========================================================================
    # TAB 3: UNIDADES
    # ========================================================================
    
    with tab3:
        st.header("📦 Análisis de Unidades")
        
        if 'unidades' in metrics:
            # Market share
            ms_cols = [col for col in df.columns if 'MS_Units_' in col]
            if ms_cols:
                st.markdown("### 📊 Market Share en Unidades")
                fig_ms_units = create_units_market_share_chart(df, analyzer.date_column)
                st.plotly_chart(fig_ms_units, width='stretch')
            
            # Métricas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Tendencia de Crecimiento")
                if 'tendencia' in metrics['unidades']:
                    tend = metrics['unidades']['tendencia']
                    st.markdown(f"**Pendiente mensual:** {tend['slope']:+,.0f} unidades/mes")
                    st.markdown(f"**R-squared:** {tend['r_squared']:.3f}")
                    
                    if tend['significant']:
                        if tend['direction'] == 'creciente':
                            st.markdown('<div class="success-box">✅ Tendencia CRECIENTE significativa</div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="danger-box">⚠️ Tendencia DECRECIENTE significativa</div>', 
                                      unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box">ℹ️ Tendencia estable</div>', 
                                  unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🎯 Market Share Actual")
                if 'ms_client' in metrics['unidades']:
                    ms = metrics['unidades']['ms_client']
                    st.markdown(f"**MS Actual:** {ms['actual']:.1f}%")
                    st.markdown(f"**Cambio:** {ms['cambio_pp']:+.1f} pp")
                    
                    if ms['cambio_pp'] > 1:
                        st.markdown('<div class="success-box">✅ Ganando market share</div>', 
                                  unsafe_allow_html=True)
                    elif ms['cambio_pp'] < -1:
                        st.markdown('<div class="danger-box">⚠️ Perdiendo market share</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box">➡️ Market share estable</div>', 
                                  unsafe_allow_html=True)
            
            # Tabla de unidades
            st.markdown("### 📋 Estadísticas de Unidades")
            
            units_data = []
            for brand, stats in metrics['unidades'].items():
                if isinstance(stats, dict) and 'promedio' in stats and 'total' in stats:
                    # Usar el nombre real de la columna
                    label = f"{brand} (Cliente)" if brand == 'client' else brand
                    units_data.append({
                        'Marca': label,
                        'Total': f"{stats['total']:,.0f}",
                        'Promedio Mensual': f"{stats['promedio']:,.0f}",
                        'Actual': f"{stats['actual']:,.0f}",
                        'Cambio': f"{stats['cambio_pct']:+.1f}%",
                        'CV': f"{stats['cv']:.1f}%"
                    })
            
            if units_data:
                units_df = pd.DataFrame(units_data)
                st.dataframe(units_df, width='stretch', hide_index=True)
    
    # ========================================================================
    # TAB 4: VALOR
    # ========================================================================
    
    with tab4:
        st.header("💵 Análisis de Ventas en Valor")
        
        if 'valor' in metrics:
            # Gráfico comparativo
            if analyzer.value_vars:
                fig_value = create_value_comparison_chart(
                    df, analyzer.date_column, analyzer.value_vars
                )
                st.plotly_chart(fig_value, width='stretch')
            
            # Métricas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'client' in metrics['valor']:
                    st.markdown("### 💰 Ventas Totales")
                    total = metrics['valor']['client']['total']
                    st.markdown(f"**Total Período:** ${total:,.0f}")
                    st.markdown(f"**Promedio Mensual:** ${metrics['valor']['client']['promedio']:,.0f}")
            
            with col2:
                if 'ms_client' in metrics['valor']:
                    st.markdown("### 📊 Market Share")
                    ms = metrics['valor']['ms_client']
                    st.markdown(f"**MS Actual:** {ms['actual']:.1f}%")
                    st.markdown(f"**Cambio:** {ms['cambio_pp']:+.1f} pp")
            
            with col3:
                if 'cagr_client' in metrics['valor']:
                    st.markdown("### 📈 Crecimiento")
                    cagr = metrics['valor']['cagr_client']
                    st.markdown(f"**CAGR Anual:** {cagr['anual']:+.1f}%")
                    
                    if cagr['anual'] > 5:
                        st.markdown("🟢 Crecimiento fuerte")
                    elif cagr['anual'] > 0:
                        st.markdown("🟡 Crecimiento moderado")
                    else:
                        st.markdown("🔴 Decrecimiento")
            
            # Tabla comparativa
            st.markdown("### 📋 Comparativa de Ventas")
            
            value_data = []
            for brand, stats in metrics['valor'].items():
                if isinstance(stats, dict) and all(k in stats for k in ['total', 'promedio', 'actual', 'cambio_pct']):
                    # Usar el nombre real de la columna
                    label = f"{brand} (Cliente)" if brand == 'client' else brand
                    value_data.append({
                        'Marca': label,
                        'Total': f"${stats['total']:,.0f}",
                        'Promedio Mensual': f"${stats['promedio']:,.0f}",
                        'Actual': f"${stats['actual']:,.0f}",
                        'Cambio': f"{stats['cambio_pct']:+.1f}%"
                    })
            
            if value_data:
                value_df = pd.DataFrame(value_data)
                st.dataframe(value_df, width='stretch', hide_index=True)
    
    # ========================================================================
    # TAB 5: DISTRIBUCIÓN
    # ========================================================================
    
    with tab5:
        st.header("🏪 Análisis de Distribución")
        
        if 'distribucion' in metrics:
            # Gráfico
            if analyzer.dist_vars:
                fig_dist = create_distribution_analysis_chart(
                    df, analyzer.date_column, analyzer.dist_vars
                )
                st.plotly_chart(fig_dist, width='stretch')
            
            # Fair Share Analysis
            st.markdown("### ⚖️ Fair Share Analysis")
            
            if 'fair_share' in metrics['distribucion']:
                fsi = metrics['distribucion']['fair_share']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Fair Share Index:** {fsi['actual']:.2f}")
                    st.markdown(f"**Promedio Período:** {fsi['promedio']:.2f}")
                
                with col2:
                    if fsi['status'] == 'over':
                        st.markdown('<div class="success-box">✅ SOBRE-PERFORMANCE<br>Ventas superan distribución</div>', 
                                  unsafe_allow_html=True)
                    elif fsi['status'] == 'under':
                        st.markdown('<div class="warning-box">⚠️ BAJO-PERFORMANCE<br>Ventas por debajo de distribución</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box">➡️ PERFORMANCE EQUILIBRADO</div>', 
                                  unsafe_allow_html=True)
            
            # Tendencia
            if 'tendencia' in metrics['distribucion']:
                st.markdown("### 📈 Tendencia de Distribución")
                tend = metrics['distribucion']['tendencia']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Pendiente mensual:** {tend['slope']:+,.1f} PDV/mes")
                
                with col2:
                    if tend['significant']:
                        if tend['direction'] == 'expansion':
                            st.markdown('<div class="success-box">✅ Expansión CRECIENTE</div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="danger-box">⚠️ Contracción</div>', 
                                      unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box">ℹ️ Distribución ESTABLE</div>', 
                                  unsafe_allow_html=True)
            
            # Tabla
            st.markdown("### 📋 Estadísticas de Distribución")
            
            dist_data = []
            for brand, stats in metrics['distribucion'].items():
                if isinstance(stats, dict) and all(k in stats for k in ['promedio', 'actual', 'cambio_pct', 'min', 'max']):
                    # Usar el nombre real de la columna
                    label = f"{brand} (Cliente)" if brand == 'client' else brand
                    dist_data.append({
                        'Marca': label,
                        'Promedio': f"{stats['promedio']:,.0f} PDV",
                        'Actual': f"{stats['actual']:,.0f} PDV",
                        'Cambio': f"{stats['cambio_pct']:+.1f}%",
                        'Min': f"{stats['min']:,.0f}",
                        'Max': f"{stats['max']:,.0f}"
                    })
            
            if dist_data:
                dist_df = pd.DataFrame(dist_data)
                st.dataframe(dist_df, width='stretch', hide_index=True)
    
    # ========================================================================
    # TAB 6: GOOGLE TRENDS
    # ========================================================================
    
    with tab6:
        st.header("🔍 Análisis de Google Trends")
        
        if 'google_trends' in metrics:
            # Gráfico
            if analyzer.gt_vars:
                fig_gt = create_google_trends_chart(
                    df, analyzer.date_column, analyzer.gt_vars
                )
                st.plotly_chart(fig_gt, width='stretch')
            
            # Métricas clave
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'client' in metrics['google_trends']:
                    st.markdown("### 📊 Interés de Búsqueda")
                    gt = metrics['google_trends']['client']
                    st.markdown(f"**Promedio:** {gt['promedio']:.1f}")
                    st.markdown(f"**Actual:** {gt['actual']:.1f}")
                    st.markdown(f"**Cambio:** {gt['cambio_pct']:+.1f}%")
            
            with col2:
                if 'sos_client' in metrics['google_trends']:
                    st.markdown("### 🎯 Share of Search")
                    sos = metrics['google_trends']['sos_client']
                    st.markdown(f"**SoS Promedio:** {sos['promedio']:.1f}%")
                    st.markdown(f"**SoS Actual:** {sos['actual']:.1f}%")
                    st.markdown(f"**Cambio:** {sos['cambio_pp']:+.1f} pp")
            
            with col3:
                if 'momentum' in metrics['google_trends']:
                    st.markdown("### 🚀 Momentum")
                    mom = metrics['google_trends']['momentum']
                    st.markdown(f"**Cambio Reciente:** {mom['valor']:+.1f}%")
                    
                    if mom['valor'] > 10:
                        st.markdown("🟢 Momentum FUERTE")
                    elif mom['valor'] > 0:
                        st.markdown("🟡 Momentum POSITIVO")
                    elif mom['valor'] > -10:
                        st.markdown("🟡 Momentum NEGATIVO")
                    else:
                        st.markdown("🔴 Momentum MUY NEGATIVO")
            
            # Correlación con ventas
            if 'correlation_sales' in metrics['google_trends']:
                st.markdown("### 🔗 Correlación Google Trends vs Ventas")
                
                corr = metrics['google_trends']['correlation_sales']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Correlación:** {corr['correlation']:.3f}")
                    st.markdown(f"**P-value:** {corr['p_value']:.4f}")
                
                with col2:
                    if corr['significant']:
                        if corr['correlation'] > 0.5:
                            st.markdown('<div class="success-box">✅ Correlación FUERTE positiva</div>', 
                                      unsafe_allow_html=True)
                        elif corr['correlation'] > 0.3:
                            st.markdown('<div class="info-box">ℹ️ Correlación MODERADA positiva</div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="info-box">ℹ️ Correlación DÉBIL</div>', 
                                      unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">⚠️ Sin relación significativa</div>', 
                                  unsafe_allow_html=True)
            
            # Tabla
            st.markdown("### 📋 Estadísticas de Google Trends")
            
            gt_data = []
            for brand, stats in metrics['google_trends'].items():
                if isinstance(stats, dict) and all(k in stats for k in ['promedio', 'actual', 'cambio_pct', 'min', 'max']):
                    # Usar el nombre real de la columna
                    label = f"{brand} (Cliente)" if brand == 'client' else brand
                    gt_data.append({
                        'Marca': label,
                        'Promedio': f"{stats['promedio']:.1f}",
                        'Actual': f"{stats['actual']:.1f}",
                        'Cambio': f"{stats['cambio_pct']:+.1f}%",
                        'Min': f"{stats['min']:.1f}",
                        'Max': f"{stats['max']:.1f}"
                    })
            
            if gt_data:
                gt_df = pd.DataFrame(gt_data)
                st.dataframe(gt_df, width='stretch', hide_index=True)
    
    # ========================================================================
    # TAB 7: DASHBOARD INTEGRADO
    # ========================================================================
    
    with tab7:
        st.header("🎯 Dashboard Integrado")
        
        st.markdown("### 🌐 Visión 360° de la Competencia")
        
        # Matriz de correlaciones
        vars_dict = {
            'precio': analyzer.price_vars,
            'unidades': analyzer.units_vars,
            'valor': analyzer.value_vars,
            'distribucion': analyzer.dist_vars,
            'google_trends': analyzer.gt_vars
        }
        
        corr_fig = create_correlation_heatmap(df, vars_dict)
        if corr_fig:
            st.plotly_chart(corr_fig, width='stretch')
        
        # Resumen de todas las dimensiones
        st.markdown("### 📊 Resumen Multi-Dimensional")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Fortalezas Competitivas")
            
            strengths = []
            
            # Revisar cada dimensión
            if 'precios' in metrics and 'positioning' in metrics['precios']:
                if metrics['precios']['positioning']['category'] == 'discount':
                    strengths.append("💰 Ventaja de precio competitivo")
            
            if 'unidades' in metrics and 'tendencia' in metrics['unidades']:
                if metrics['unidades']['tendencia']['direction'] == 'creciente' and \
                   metrics['unidades']['tendencia']['significant']:
                    strengths.append("📈 Crecimiento sostenido en unidades")
            
            if 'valor' in metrics and 'ms_client' in metrics['valor']:
                if metrics['valor']['ms_client']['cambio_pp'] > 0:
                    strengths.append("🎯 Ganancia de market share en valor")
            
            if 'distribucion' in metrics and 'fair_share' in metrics['distribucion']:
                if metrics['distribucion']['fair_share']['status'] == 'over':
                    strengths.append("✅ Sobre-performance en ventas vs distribución")
            
            if 'google_trends' in metrics and 'momentum' in metrics['google_trends']:
                if metrics['google_trends']['momentum']['valor'] > 10:
                    strengths.append("🚀 Momentum fuerte en interés de búsqueda")
            
            if strengths:
                for strength in strengths:
                    st.markdown(f'<div class="success-box">{strength}</div>', unsafe_allow_html=True)
            else:
                st.info("No se detectaron fortalezas significativas")
        
        with col2:
            st.markdown("#### ⚠️ Áreas de Oportunidad")
            
            opportunities = []
            
            # Revisar cada dimensión
            if 'precios' in metrics and 'positioning' in metrics['precios']:
                if metrics['precios']['positioning']['premium_index'] > 20:
                    opportunities.append("💰 Precio significativamente más alto que mercado")
            
            if 'unidades' in metrics and 'tendencia' in metrics['unidades']:
                if metrics['unidades']['tendencia']['direction'] == 'decreciente' and \
                   metrics['unidades']['tendencia']['significant']:
                    opportunities.append("📉 Tendencia negativa en unidades")
            
            if 'valor' in metrics and 'ms_client' in metrics['valor']:
                if metrics['valor']['ms_client']['cambio_pp'] < -1:
                    opportunities.append("⚠️ Pérdida de market share en valor")
            
            if 'distribucion' in metrics and 'fair_share' in metrics['distribucion']:
                if metrics['distribucion']['fair_share']['status'] == 'under':
                    opportunities.append("🏪 Bajo-performance vs distribución")
            
            if 'google_trends' in metrics and 'momentum' in metrics['google_trends']:
                if metrics['google_trends']['momentum']['valor'] < -10:
                    opportunities.append("🔍 Momentum negativo en búsquedas")
            
            if opportunities:
                for opp in opportunities:
                    st.markdown(f'<div class="warning-box">{opp}</div>', unsafe_allow_html=True)
            else:
                st.success("No se detectaron áreas críticas de mejora")
        
        # Recomendaciones estratégicas
        st.markdown("---")
        st.markdown("### 💡 Recomendaciones Estratégicas")
        
        recommendations = []
        
        # Basadas en el análisis
        if 'precios' in metrics and 'positioning' in metrics['precios']:
            premium_idx = metrics['precios']['positioning']['premium_index']
            if premium_idx > 20 and 'unidades' in metrics:
                if metrics['unidades'].get('ms_client', {}).get('cambio_pp', 0) < 0:
                    recommendations.append(
                        "🎯 **Estrategia de Precio:** Considerar ajuste de precio para mejorar competitividad. "
                        "El premium significativo puede estar afectando participación de mercado."
                    )
        
        if 'distribucion' in metrics and 'fair_share' in metrics['distribucion']:
            if metrics['distribucion']['fair_share']['status'] == 'under':
                recommendations.append(
                    "🏪 **Estrategia de Trade:** Mejorar activación en punto de venta. "
                    "La distribución no se está traduciendo eficientemente en ventas."
                )
        
        if 'google_trends' in metrics and 'correlation_sales' in metrics['google_trends']:
            if metrics['google_trends']['correlation_sales']['significant'] and \
               metrics['google_trends']['correlation_sales']['correlation'] > 0.5:
                if metrics['google_trends'].get('momentum', {}).get('valor', 0) < 0:
                    recommendations.append(
                        "📱 **Estrategia Digital:** Incrementar inversión en marketing digital. "
                        "Existe alta correlación entre búsquedas y ventas, y el momentum es negativo."
                    )
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f'<div class="info-box"><strong>{i}.</strong> {rec}</div>', 
                          unsafe_allow_html=True)
        else:
            st.info("Mantener estrategia actual y monitorear evolución del mercado")

else:
    # ========================================================================
    # PANTALLA DE BIENVENIDA
    # ========================================================================
    
    st.markdown("""
    ## 👋 Bienvenido al Competitive Analyzer
    
    Esta herramienta te ayuda a analizar tu posición competitiva en **5 dimensiones clave**:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Dimensiones de Análisis:
        
        - 💰 **Precios** - Posicionamiento y elasticidad
        - 📦 **Unidades** - Volumen y market share
        - 💵 **Valor** - Ventas y participación
        - 🏪 **Distribución** - Cobertura y eficiencia
        - 🔍 **Google Trends** - Interés de búsqueda
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Cómo Empezar:
        
        1. **Sube tu archivo** Excel (panel izquierdo)
        2. **Selecciona las variables** para cada dimensión
           - Variable del cliente
           - Variables de competidores
        3. **Ejecuta** el análisis completo
        4. **Explora** los insights interactivos
        5. **Toma decisiones** basadas en datos
        """)
    
    st.markdown("---")
    
    # Ejemplo de estructura de datos
    st.markdown("### 📋 Estructura de Datos Esperada:")
    
    st.code("""
    Columnas requeridas:
    - Date: Fecha (formato YYYY-MM-DD o MM/YYYY)
    - Precio_[Marca]: Precio promedio por marca
    - Unid_[Marca]: Unidades vendidas por marca
    - SalesValue_[Marca] o Value_[Marca]: Ventas en valor
    - Dist_[Marca]: Distribución numérica (PDV)
    - [Marca]_GT: Google Trends (opcional)
    
    Ejemplo:
    Date, Precio_LIST, Unid_LIST, SalesValue_LIST, Dist_LIST, LIS_GT,
          Precio_Colgate, Unid_Colgate, Value_Colgate, Dist_COLGATE, COLGENJ_GT
    """, language="text")
    
    st.markdown("---")
    
    # FAQs
    with st.expander("❓ Preguntas Frecuentes"):
        st.markdown("""
        **¿Qué es el Fair Share Index?**
        
        Mide si tus ventas están en línea con tu distribución. FSI > 1 indica sobre-performance
        (ventas mayores que lo esperado por distribución). FSI < 1 indica bajo-performance.
        
        **¿Cómo se calcula el Market Share?**
        
        MS = (Ventas de tu marca / Total ventas del mercado) × 100
        Se calcula tanto para unidades como para valor.
        
        **¿Qué es el Share of Search?**
        
        Es tu participación en el total de búsquedas de Google de la categoría.
        Un indicador líder de interés del consumidor.
        
        **¿Qué significa el Momentum?**
        
        Compara el promedio de los últimos 3 meses vs los 3 meses anteriores.
        Indica si estás ganando o perdiendo tracción.
        
        **¿Cuántos competidores puedo analizar?**
        
        No hay límite, pero recomendamos enfocarse en los 3-5 principales competidores
        para mantener el análisis manejable y accionable.
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Competitive Analyzer v1.0</strong></p>
        <p>Multi-Dimensional Competitive Intelligence Tool</p>
        <p>Desarrollado por el Equipo de Analytics</p>
    </div>
    """, unsafe_allow_html=True)
