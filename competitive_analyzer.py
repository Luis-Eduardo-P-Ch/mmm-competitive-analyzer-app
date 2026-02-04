"""
Competitive Analyzer - Análisis Competitivo Multi-Dimensional
==============================================================
Herramienta profesional para análisis competitivo de precios, unidades,
ventas, distribución y Google Trends

Autor: Analytics Team
Versión: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class CompetitiveAnalyzer:
    """
    Analizador competitivo multi-dimensional para análisis de mercado
    """
    
    def __init__(self, filepath, date_column='Date'):
        """
        Inicializa el analizador competitivo
        
        Parameters:
        -----------
        filepath : str
            Ruta al archivo Excel
        date_column : str
            Nombre de la columna de fecha
        """
        self.filepath = filepath
        self.date_column = date_column
        self.df = None
        self.insights = []
        self.warnings_list = []
        self.metrics = {}
        
        # Variables por dimensión
        self.price_vars = {}
        self.units_vars = {}
        self.value_vars = {}
        self.dist_vars = {}
        self.gt_vars = {}
        
        self._load_data()
        
    def _load_data(self):
        """Carga los datos desde Excel"""
        print("📊 Cargando datos...")
        self.df = pd.read_excel(self.filepath)
        
        # Limpiar datos
        self.df = self.df.dropna(how='all')
        
        # Convertir fecha
        if self.date_column in self.df.columns:
            self.df[self.date_column] = pd.to_datetime(self.df[self.date_column], errors='coerce')
            self.df = self.df.dropna(subset=[self.date_column])
            self.df = self.df.sort_values(self.date_column).reset_index(drop=True)
        
        print(f"✅ Datos cargados: {self.df.shape[0]} períodos, {self.df.shape[1]} variables")
        
    def configure_variables(self, client_brand, competitors):
        """
        Configura las variables para análisis
        
        Parameters:
        -----------
        client_brand : str
            Nombre de la marca cliente (ej: 'LIST', 'Listerine')
        competitors : list
            Lista de competidores (ej: ['Colgate', 'OralB'])
        """
        self.client_brand = client_brand
        self.competitors = competitors
        
        # Detectar automáticamente variables por dimensión
        self._detect_variables()
        
    def _detect_variables(self):
        """Detecta automáticamente las variables disponibles"""
        columns = self.df.columns.tolist()
        
        # PRECIOS
        for col in columns:
            col_upper = col.upper()
            if 'PRECIO' in col_upper or 'PRICE' in col_upper:
                if self.client_brand.upper() in col_upper:
                    self.price_vars['client'] = col
                else:
                    for comp in self.competitors:
                        if comp.upper() in col_upper:
                            self.price_vars[comp] = col
        
        # UNIDADES
        for col in columns:
            col_upper = col.upper()
            if 'UNID' in col_upper or 'UNIT' in col_upper or 'QTY' in col_upper:
                if self.client_brand.upper() in col_upper:
                    self.units_vars['client'] = col
                else:
                    for comp in self.competitors:
                        if comp.upper() in col_upper:
                            self.units_vars[comp] = col
        
        # VALOR (SALES VALUE)
        for col in columns:
            col_upper = col.upper()
            if ('VALUE' in col_upper or 'SALES' in col_upper or 'VALOR' in col_upper) and 'SALESVALUE' in col_upper:
                if self.client_brand.upper() in col_upper:
                    self.value_vars['client'] = col
                else:
                    for comp in self.competitors:
                        if comp.upper() in col_upper:
                            self.value_vars[comp] = col
        
        # DISTRIBUCIÓN
        for col in columns:
            col_upper = col.upper()
            if 'DIST' in col_upper and 'SALESVALUE' not in col_upper:
                if self.client_brand.upper() in col_upper:
                    self.dist_vars['client'] = col
                else:
                    for comp in self.competitors:
                        if comp.upper() in col_upper:
                            self.dist_vars[comp] = col
        
        # GOOGLE TRENDS (GT)
        for col in columns:
            col_upper = col.upper()
            if '_GT' in col_upper or 'GOOGLE' in col_upper or 'TREND' in col_upper:
                if self.client_brand.upper() in col_upper:
                    self.gt_vars['client'] = col
                else:
                    for comp in self.competitors:
                        if comp.upper() in col_upper:
                            self.gt_vars[comp] = col
        
        print(f"\n✅ Variables detectadas:")
        print(f"   Precios: {len(self.price_vars)}")
        print(f"   Unidades: {len(self.units_vars)}")
        print(f"   Valor: {len(self.value_vars)}")
        print(f"   Distribución: {len(self.dist_vars)}")
        print(f"   Google Trends: {len(self.gt_vars)}")
    
    def analyze_prices(self):
        """Análisis detallado de precios"""
        print("\n" + "="*80)
        print("💰 ANÁLISIS DE PRECIOS")
        print("="*80)
        
        if not self.price_vars:
            print("⚠️ No se encontraron variables de precio")
            return {}
        
        price_analysis = {}
        
        # Precio promedio
        print("\n📊 Precios Promedio:")
        for brand, col in self.price_vars.items():
            if col in self.df.columns:
                avg_price = self.df[col].mean()
                current_price = self.df[col].iloc[-1]
                initial_price = self.df[col].iloc[0]
                price_change = ((current_price - initial_price) / initial_price * 100)
                
                label = "Cliente" if brand == 'client' else brand
                print(f"   {label}: ${avg_price:,.0f} | Actual: ${current_price:,.0f} | Cambio: {price_change:+.1f}%")
                
                price_analysis[brand] = {
                    'promedio': avg_price,
                    'actual': current_price,
                    'inicial': initial_price,
                    'cambio_pct': price_change,
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'std': self.df[col].std(),
                    'cv': (self.df[col].std() / self.df[col].mean() * 100)
                }
        
        # Índice de precio relativo (Cliente vs Competencia)
        if 'client' in self.price_vars:
            client_col = self.price_vars['client']
            print("\n📈 Índice de Precio Relativo (Cliente vs Competencia):")
            
            for brand, col in self.price_vars.items():
                if brand != 'client' and col in self.df.columns:
                    self.df[f'PriceIndex_{brand}'] = (self.df[client_col] / self.df[col] * 100)
                    avg_index = self.df[f'PriceIndex_{brand}'].mean()
                    current_index = self.df[f'PriceIndex_{brand}'].iloc[-1]
                    
                    status = "🟢 Más bajo" if avg_index < 100 else "🔴 Más alto"
                    print(f"   vs {brand}: {avg_index:.1f} | Actual: {current_index:.1f} | {status}")
                    
                    price_analysis[f'index_vs_{brand}'] = {
                        'promedio': avg_index,
                        'actual': current_index,
                        'tendencia': 'favorable' if avg_index < 100 else 'desfavorable'
                    }
            
            # Premium/Discount positioning
            avg_comp_prices = []
            for brand, col in self.price_vars.items():
                if brand != 'client' and col in self.df.columns:
                    avg_comp_prices.append(self.df[col].mean())
            
            if avg_comp_prices:
                avg_market_price = np.mean(avg_comp_prices)
                client_avg = self.df[client_col].mean()
                premium_index = (client_avg / avg_market_price - 1) * 100
                
                print(f"\n💎 Posicionamiento de Precio:")
                print(f"   Precio Cliente: ${client_avg:,.0f}")
                print(f"   Precio Mercado: ${avg_market_price:,.0f}")
                print(f"   Premium/Discount: {premium_index:+.1f}%")
                
                if premium_index > 10:
                    print(f"   ➡️ Posicionamiento PREMIUM")
                elif premium_index < -10:
                    print(f"   ➡️ Posicionamiento DISCOUNT")
                else:
                    print(f"   ➡️ Posicionamiento AT PAR")
                
                price_analysis['positioning'] = {
                    'premium_index': premium_index,
                    'category': 'premium' if premium_index > 10 else 'discount' if premium_index < -10 else 'at_par'
                }
        
        # Elasticidad precio (correlación precio vs unidades)
        if 'client' in self.price_vars and 'client' in self.units_vars:
            price_col = self.price_vars['client']
            units_col = self.units_vars['client']
            
            if price_col in self.df.columns and units_col in self.df.columns:
                corr, pval = pearsonr(self.df[price_col].dropna(), 
                                     self.df[units_col].dropna())
                
                print(f"\n🔗 Elasticidad Precio-Unidades:")
                print(f"   Correlación: {corr:.3f} (p-value: {pval:.4f})")
                
                if pval < 0.05:
                    if corr < -0.3:
                        print(f"   ➡️ Elasticidad NEGATIVA significativa (↑precio = ↓unidades)")
                    elif corr > 0.3:
                        print(f"   ➡️ Relación POSITIVA (posible efecto calidad/premium)")
                    else:
                        print(f"   ➡️ Elasticidad DÉBIL")
                else:
                    print(f"   ➡️ No hay relación significativa")
                
                price_analysis['elasticity'] = {
                    'correlation': corr,
                    'p_value': pval,
                    'significant': pval < 0.05
                }
        
        self.metrics['precios'] = price_analysis
        return price_analysis
    
    def analyze_units(self):
        """Análisis detallado de unidades vendidas"""
        print("\n" + "="*80)
        print("📦 ANÁLISIS DE UNIDADES VENDIDAS")
        print("="*80)
        
        if not self.units_vars:
            print("⚠️ No se encontraron variables de unidades")
            return {}
        
        units_analysis = {}
        
        # Unidades totales y promedios
        print("\n📊 Unidades Promedio Mensuales:")
        for brand, col in self.units_vars.items():
            if col in self.df.columns:
                total_units = self.df[col].sum()
                avg_units = self.df[col].mean()
                current_units = self.df[col].iloc[-1]
                initial_units = self.df[col].iloc[0]
                units_change = ((current_units - initial_units) / initial_units * 100)
                
                label = "Cliente" if brand == 'client' else brand
                print(f"   {label}: {avg_units:,.0f} | Total: {total_units:,.0f} | Cambio: {units_change:+.1f}%")
                
                units_analysis[brand] = {
                    'promedio': avg_units,
                    'total': total_units,
                    'actual': current_units,
                    'inicial': initial_units,
                    'cambio_pct': units_change,
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'std': self.df[col].std(),
                    'cv': (self.df[col].std() / self.df[col].mean() * 100)
                }
        
        # Market Share de unidades
        if len(self.units_vars) > 1:
            print("\n📈 Market Share en Unidades:")
            
            # Crear columna de total mercado
            units_cols = [col for col in self.units_vars.values() if col in self.df.columns]
            self.df['Total_Market_Units'] = self.df[units_cols].sum(axis=1)
            
            for brand, col in self.units_vars.items():
                if col in self.df.columns:
                    ms_col = f'MS_Units_{brand}'
                    self.df[ms_col] = (self.df[col] / self.df['Total_Market_Units'] * 100)
                    
                    avg_ms = self.df[ms_col].mean()
                    current_ms = self.df[ms_col].iloc[-1]
                    initial_ms = self.df[ms_col].iloc[0]
                    ms_change = current_ms - initial_ms
                    
                    label = "Cliente" if brand == 'client' else brand
                    print(f"   {label}: {avg_ms:.1f}% | Actual: {current_ms:.1f}% | Cambio: {ms_change:+.1f} pp")
                    
                    units_analysis[f'ms_{brand}'] = {
                        'promedio': avg_ms,
                        'actual': current_ms,
                        'inicial': initial_ms,
                        'cambio_pp': ms_change
                    }
        
        # Tendencia de crecimiento
        if 'client' in self.units_vars:
            client_col = self.units_vars['client']
            if client_col in self.df.columns:
                # Regresión lineal para tendencia
                x = np.arange(len(self.df))
                y = self.df[client_col].values
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                print(f"\n📈 Tendencia de Unidades (Cliente):")
                print(f"   Pendiente mensual: {slope:+,.0f} unidades/mes")
                print(f"   R-squared: {r_value**2:.3f}")
                
                if p_value < 0.05:
                    if slope > 0:
                        print(f"   ➡️ Tendencia CRECIENTE significativa ✅")
                    else:
                        print(f"   ➡️ Tendencia DECRECIENTE significativa ⚠️")
                else:
                    print(f"   ➡️ Tendencia NO significativa (estable)")
                
                units_analysis['tendencia'] = {
                    'slope': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'direction': 'creciente' if slope > 0 else 'decreciente',
                    'significant': p_value < 0.05
                }
        
        # Volatilidad comparativa
        print(f"\n📊 Volatilidad (Coeficiente de Variación):")
        for brand, col in self.units_vars.items():
            if col in self.df.columns:
                cv = (self.df[col].std() / self.df[col].mean() * 100)
                label = "Cliente" if brand == 'client' else brand
                status = "🟢 Baja" if cv < 20 else "🟡 Media" if cv < 40 else "🔴 Alta"
                print(f"   {label}: {cv:.1f}% | {status}")
        
        self.metrics['unidades'] = units_analysis
        return units_analysis
    
    def analyze_value(self):
        """Análisis detallado de ventas en valor"""
        print("\n" + "="*80)
        print("💵 ANÁLISIS DE VENTAS EN VALOR")
        print("="*80)
        
        if not self.value_vars:
            print("⚠️ No se encontraron variables de valor")
            return {}
        
        value_analysis = {}
        
        # Ventas totales y promedios
        print("\n📊 Ventas Promedio Mensuales:")
        for brand, col in self.value_vars.items():
            if col in self.df.columns:
                total_value = self.df[col].sum()
                avg_value = self.df[col].mean()
                current_value = self.df[col].iloc[-1]
                initial_value = self.df[col].iloc[0]
                value_change = ((current_value - initial_value) / initial_value * 100)
                
                label = "Cliente" if brand == 'client' else brand
                print(f"   {label}: ${avg_value:,.0f} | Total: ${total_value:,.0f} | Cambio: {value_change:+.1f}%")
                
                value_analysis[brand] = {
                    'promedio': avg_value,
                    'total': total_value,
                    'actual': current_value,
                    'inicial': initial_value,
                    'cambio_pct': value_change,
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'std': self.df[col].std()
                }
        
        # Market Share de valor
        if len(self.value_vars) > 1:
            print("\n📈 Market Share en Valor:")
            
            # Crear columna de total mercado
            value_cols = [col for col in self.value_vars.values() if col in self.df.columns]
            self.df['Total_Market_Value'] = self.df[value_cols].sum(axis=1)
            
            for brand, col in self.value_vars.items():
                if col in self.df.columns:
                    ms_col = f'MS_Value_{brand}'
                    self.df[ms_col] = (self.df[col] / self.df['Total_Market_Value'] * 100)
                    
                    avg_ms = self.df[ms_col].mean()
                    current_ms = self.df[ms_col].iloc[-1]
                    initial_ms = self.df[ms_col].iloc[0]
                    ms_change = current_ms - initial_ms
                    
                    label = "Cliente" if brand == 'client' else brand
                    print(f"   {label}: {avg_ms:.1f}% | Actual: {current_ms:.1f}% | Cambio: {ms_change:+.1f} pp")
                    
                    value_analysis[f'ms_{brand}'] = {
                        'promedio': avg_ms,
                        'actual': current_ms,
                        'inicial': initial_ms,
                        'cambio_pp': ms_change
                    }
        
        # Valor por unidad (precio promedio efectivo)
        if 'client' in self.value_vars and 'client' in self.units_vars:
            value_col = self.value_vars['client']
            units_col = self.units_vars['client']
            
            if value_col in self.df.columns and units_col in self.df.columns:
                self.df['Avg_Unit_Value'] = self.df[value_col] / self.df[units_col]
                avg_unit_value = self.df['Avg_Unit_Value'].mean()
                
                print(f"\n💰 Valor Promedio por Unidad (Cliente): ${avg_unit_value:,.0f}")
                
                value_analysis['avg_unit_value'] = avg_unit_value
        
        # Growth rate comparison
        print(f"\n📈 Tasa de Crecimiento (CAGR Anualizado):")
        for brand, col in self.value_vars.items():
            if col in self.df.columns:
                n_periods = len(self.df)
                if n_periods > 1:
                    initial = self.df[col].iloc[0]
                    final = self.df[col].iloc[-1]
                    
                    if initial > 0:
                        # CAGR mensual
                        cagr_monthly = ((final / initial) ** (1/n_periods) - 1) * 100
                        # Anualizado (asumiendo datos mensuales)
                        cagr_annual = ((1 + cagr_monthly/100) ** 12 - 1) * 100
                        
                        label = "Cliente" if brand == 'client' else brand
                        status = "🟢" if cagr_annual > 5 else "🟡" if cagr_annual > 0 else "🔴"
                        print(f"   {label}: {cagr_annual:+.1f}% anual | {status}")
                        
                        value_analysis[f'cagr_{brand}'] = {
                            'mensual': cagr_monthly,
                            'anual': cagr_annual
                        }
        
        self.metrics['valor'] = value_analysis
        return value_analysis
    
    def analyze_distribution(self):
        """Análisis detallado de distribución"""
        print("\n" + "="*80)
        print("🏪 ANÁLISIS DE DISTRIBUCIÓN")
        print("="*80)
        
        if not self.dist_vars:
            print("⚠️ No se encontraron variables de distribución")
            return {}
        
        dist_analysis = {}
        
        # Distribución promedio y evolución
        print("\n📊 Distribución Numérica Promedio:")
        for brand, col in self.dist_vars.items():
            if col in self.df.columns:
                avg_dist = self.df[col].mean()
                current_dist = self.df[col].iloc[-1]
                initial_dist = self.df[col].iloc[0]
                dist_change = ((current_dist - initial_dist) / initial_dist * 100) if initial_dist > 0 else 0
                
                label = "Cliente" if brand == 'client' else brand
                print(f"   {label}: {avg_dist:,.0f} PDV | Actual: {current_dist:,.0f} | Cambio: {dist_change:+.1f}%")
                
                dist_analysis[brand] = {
                    'promedio': avg_dist,
                    'actual': current_dist,
                    'inicial': initial_dist,
                    'cambio_pct': dist_change,
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'std': self.df[col].std()
                }
        
        # Share de distribución
        if len(self.dist_vars) > 1:
            print("\n📈 Share de Distribución:")
            
            # Crear columna de total mercado
            dist_cols = [col for col in self.dist_vars.values() if col in self.df.columns]
            self.df['Total_Market_Dist'] = self.df[dist_cols].sum(axis=1)
            
            for brand, col in self.dist_vars.items():
                if col in self.df.columns:
                    sd_col = f'SD_{brand}'
                    self.df[sd_col] = (self.df[col] / self.df['Total_Market_Dist'] * 100)
                    
                    avg_sd = self.df[sd_col].mean()
                    current_sd = self.df[sd_col].iloc[-1]
                    initial_sd = self.df[sd_col].iloc[0]
                    sd_change = current_sd - initial_sd
                    
                    label = "Cliente" if brand == 'client' else brand
                    print(f"   {label}: {avg_sd:.1f}% | Actual: {current_sd:.1f}% | Cambio: {sd_change:+.1f} pp")
                    
                    dist_analysis[f'share_{brand}'] = {
                        'promedio': avg_sd,
                        'actual': current_sd,
                        'inicial': initial_sd,
                        'cambio_pp': sd_change
                    }
        
        # Fair Share Analysis (distribución vs ventas)
        if 'client' in self.dist_vars and 'client' in self.value_vars:
            dist_col = self.dist_vars['client']
            value_col = self.value_vars['client']
            
            if dist_col in self.df.columns and value_col in self.df.columns:
                if 'Total_Market_Dist' in self.df.columns and 'Total_Market_Value' in self.df.columns:
                    self.df['Dist_Share'] = (self.df[dist_col] / self.df['Total_Market_Dist'] * 100)
                    self.df['Value_Share'] = (self.df[value_col] / self.df['Total_Market_Value'] * 100)
                    self.df['Fair_Share_Index'] = self.df['Value_Share'] / self.df['Dist_Share']
                    
                    avg_fsi = self.df['Fair_Share_Index'].mean()
                    current_fsi = self.df['Fair_Share_Index'].iloc[-1]
                    
                    print(f"\n⚖️ Fair Share Index (Cliente):")
                    print(f"   FSI Promedio: {avg_fsi:.2f}")
                    print(f"   FSI Actual: {current_fsi:.2f}")
                    
                    if avg_fsi > 1.1:
                        print(f"   ➡️ SOBRE-PERFORMANCE (ventas > distribución) ✅")
                    elif avg_fsi < 0.9:
                        print(f"   ➡️ BAJO-PERFORMANCE (ventas < distribución) ⚠️")
                    else:
                        print(f"   ➡️ PERFORMANCE EQUILIBRADO")
                    
                    dist_analysis['fair_share'] = {
                        'promedio': avg_fsi,
                        'actual': current_fsi,
                        'status': 'over' if avg_fsi > 1.1 else 'under' if avg_fsi < 0.9 else 'balanced'
                    }
        
        # Tendencia de distribución
        if 'client' in self.dist_vars:
            client_col = self.dist_vars['client']
            if client_col in self.df.columns:
                x = np.arange(len(self.df))
                y = self.df[client_col].values
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                print(f"\n📈 Tendencia de Distribución (Cliente):")
                print(f"   Pendiente mensual: {slope:+,.1f} PDV/mes")
                
                if p_value < 0.05:
                    if slope > 0:
                        print(f"   ➡️ Expansión CRECIENTE ✅")
                    else:
                        print(f"   ➡️ Contracción ⚠️")
                else:
                    print(f"   ➡️ Distribución ESTABLE")
                
                dist_analysis['tendencia'] = {
                    'slope': slope,
                    'p_value': p_value,
                    'direction': 'expansion' if slope > 0 else 'contraction',
                    'significant': p_value < 0.05
                }
        
        self.metrics['distribucion'] = dist_analysis
        return dist_analysis
    
    def analyze_google_trends(self):
        """Análisis detallado de Google Trends"""
        print("\n" + "="*80)
        print("🔍 ANÁLISIS DE GOOGLE TRENDS")
        print("="*80)
        
        if not self.gt_vars:
            print("⚠️ No se encontraron variables de Google Trends")
            return {}
        
        gt_analysis = {}
        
        # Interés promedio
        print("\n📊 Interés de Búsqueda Promedio:")
        for brand, col in self.gt_vars.items():
            if col in self.df.columns:
                avg_gt = self.df[col].mean()
                current_gt = self.df[col].iloc[-1]
                initial_gt = self.df[col].iloc[0]
                gt_change = ((current_gt - initial_gt) / initial_gt * 100) if initial_gt > 0 else 0
                
                label = "Cliente" if brand == 'client' else brand
                print(f"   {label}: {avg_gt:.1f} | Actual: {current_gt:.1f} | Cambio: {gt_change:+.1f}%")
                
                gt_analysis[brand] = {
                    'promedio': avg_gt,
                    'actual': current_gt,
                    'inicial': initial_gt,
                    'cambio_pct': gt_change,
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'std': self.df[col].std()
                }
        
        # Share of Search
        if len(self.gt_vars) > 1:
            print("\n📈 Share of Search (SoS):")
            
            # Crear columna de total de búsquedas
            gt_cols = [col for col in self.gt_vars.values() if col in self.df.columns]
            self.df['Total_Search'] = self.df[gt_cols].sum(axis=1)
            
            for brand, col in self.gt_vars.items():
                if col in self.df.columns:
                    sos_col = f'SoS_{brand}'
                    self.df[sos_col] = (self.df[col] / self.df['Total_Search'] * 100)
                    
                    avg_sos = self.df[sos_col].mean()
                    current_sos = self.df[sos_col].iloc[-1]
                    initial_sos = self.df[sos_col].iloc[0]
                    sos_change = current_sos - initial_sos
                    
                    label = "Cliente" if brand == 'client' else brand
                    print(f"   {label}: {avg_sos:.1f}% | Actual: {current_sos:.1f}% | Cambio: {sos_change:+.1f} pp")
                    
                    gt_analysis[f'sos_{brand}'] = {
                        'promedio': avg_sos,
                        'actual': current_sos,
                        'inicial': initial_sos,
                        'cambio_pp': sos_change
                    }
        
        # Correlación Google Trends vs Ventas
        if 'client' in self.gt_vars and 'client' in self.value_vars:
            gt_col = self.gt_vars['client']
            value_col = self.value_vars['client']
            
            if gt_col in self.df.columns and value_col in self.df.columns:
                corr, pval = pearsonr(self.df[gt_col].dropna(), 
                                     self.df[value_col].dropna())
                
                print(f"\n🔗 Correlación Google Trends vs Ventas (Cliente):")
                print(f"   Correlación: {corr:.3f} (p-value: {pval:.4f})")
                
                if pval < 0.05:
                    if corr > 0.5:
                        print(f"   ➡️ Correlación FUERTE positiva ✅")
                    elif corr > 0.3:
                        print(f"   ➡️ Correlación MODERADA positiva")
                    else:
                        print(f"   ➡️ Correlación DÉBIL")
                else:
                    print(f"   ➡️ No hay relación significativa")
                
                gt_analysis['correlation_sales'] = {
                    'correlation': corr,
                    'p_value': pval,
                    'significant': pval < 0.05
                }
        
        # Momentum de búsqueda (tendencia reciente)
        if 'client' in self.gt_vars:
            client_col = self.gt_vars['client']
            if client_col in self.df.columns and len(self.df) >= 6:
                # Comparar últimos 3 meses vs 3 meses anteriores
                recent_avg = self.df[client_col].iloc[-3:].mean()
                previous_avg = self.df[client_col].iloc[-6:-3].mean()
                
                if previous_avg > 0:
                    momentum = ((recent_avg - previous_avg) / previous_avg * 100)
                    
                    print(f"\n🚀 Momentum de Búsqueda (últimos 3 meses vs anteriores):")
                    print(f"   Cambio: {momentum:+.1f}%")
                    
                    if momentum > 10:
                        print(f"   ➡️ Momentum POSITIVO fuerte ✅")
                    elif momentum > 0:
                        print(f"   ➡️ Momentum POSITIVO moderado")
                    elif momentum > -10:
                        print(f"   ➡️ Momentum NEGATIVO moderado")
                    else:
                        print(f"   ➡️ Momentum NEGATIVO fuerte ⚠️")
                    
                    gt_analysis['momentum'] = {
                        'valor': momentum,
                        'reciente': recent_avg,
                        'anterior': previous_avg
                    }
        
        self.metrics['google_trends'] = gt_analysis
        return gt_analysis
    
    def generate_competitive_insights(self):
        """Genera insights competitivos integrados"""
        print("\n" + "="*80)
        print("💡 INSIGHTS COMPETITIVOS INTEGRADOS")
        print("="*80)
        
        insights = []
        
        # Insight 1: Posición general competitiva
        if self.metrics:
            print("\n🎯 Posición Competitiva General:")
            
            # Contar métricas favorables
            favorable_count = 0
            total_metrics = 0
            
            # Precio
            if 'precios' in self.metrics and 'positioning' in self.metrics['precios']:
                total_metrics += 1
                if self.metrics['precios']['positioning']['category'] != 'premium':
                    favorable_count += 1
                    print("   ✅ Precio competitivo en mercado")
            
            # Unidades - tendencia
            if 'unidades' in self.metrics and 'tendencia' in self.metrics['unidades']:
                total_metrics += 1
                if self.metrics['unidades']['tendencia']['direction'] == 'creciente':
                    favorable_count += 1
                    print("   ✅ Crecimiento en unidades")
            
            # Valor - market share
            if 'valor' in self.metrics and 'ms_client' in self.metrics['valor']:
                total_metrics += 1
                if self.metrics['valor']['ms_client']['cambio_pp'] > 0:
                    favorable_count += 1
                    print("   ✅ Ganancia de market share en valor")
            
            # Distribución - fair share
            if 'distribucion' in self.metrics and 'fair_share' in self.metrics['distribucion']:
                total_metrics += 1
                if self.metrics['distribucion']['fair_share']['status'] == 'over':
                    favorable_count += 1
                    print("   ✅ Sobre-performance en distribución")
            
            # Google Trends
            if 'google_trends' in self.metrics and 'momentum' in self.metrics['google_trends']:
                total_metrics += 1
                if self.metrics['google_trends']['momentum']['valor'] > 0:
                    favorable_count += 1
                    print("   ✅ Momentum positivo en búsquedas")
            
            if total_metrics > 0:
                score = (favorable_count / total_metrics * 100)
                print(f"\n   📊 Score Competitivo: {score:.0f}% ({favorable_count}/{total_metrics} métricas favorables)")
                
                if score >= 70:
                    print("   ➡️ POSICIÓN COMPETITIVA FUERTE ✅")
                elif score >= 50:
                    print("   ➡️ POSICIÓN COMPETITIVA MODERADA")
                else:
                    print("   ➡️ POSICIÓN COMPETITIVA DÉBIL - Requiere acción ⚠️")
        
        return insights
    
    def run_full_analysis(self):
        """Ejecuta análisis completo de todas las dimensiones"""
        print("\n" + "🚀"*40)
        print("INICIANDO ANÁLISIS COMPETITIVO COMPLETO")
        print("🚀"*40)
        
        # Ejecutar análisis por dimensión
        self.analyze_prices()
        self.analyze_units()
        self.analyze_value()
        self.analyze_distribution()
        self.analyze_google_trends()
        
        # Insights integrados
        self.generate_competitive_insights()
        
        print("\n" + "✅"*40)
        print("ANÁLISIS COMPLETADO")
        print("✅"*40)
        
        return self.metrics
