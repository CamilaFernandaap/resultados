import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from numba import njit
from shapely.geometry import Point
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import os
import pyarrow.parquet as pq
import gc

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
    "axes.facecolor": "#f8f9fa",
    "figure.facecolor": "white",
    "grid.alpha": 0.3
})

CORES = {
    "serie": "#4A90E2",
    "tendencia": "#E74C3C",
    "filtrada": "#2C3E50",
    "positivo": "#E74C3C",
    "negativo": "#3498DB",
    "grid": "#E0E0E0",
    "temp": "#E74C3C",
    "rad": "#FFA500",
    "prec": "#3498DB"
}

@njit(fastmath=True)
def mk_sen_fast(v):
    """Mann-Kendall e Sen's slope otimizado com Numba"""
    n0 = v.size
    cnt = 0
    for i in range(n0):
        if not np.isnan(v[i]):
            cnt += 1
    if cnt < 10:
        return np.nan, np.nan
    x = np.empty(cnt)
    k = 0
    for i in range(n0):
        if not np.isnan(v[i]):
            x[k] = v[i]
            k += 1
    n = cnt
    nsl = n * (n - 1) // 2
    slopes = np.empty(nsl)
    s = 0
    idx = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = x[j] - x[i]
            slopes[idx] = d / (j - i)
            idx += 1
            if d > 0:
                s += 1
            elif d < 0:
                s -= 1
    slopes.sort()
    slope = slopes[nsl // 2] if nsl % 2 else 0.5 * (slopes[nsl // 2 - 1] + slopes[nsl // 2])
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2.0 * (1.0 - 0.5 * (1.0 + np.tanh(np.abs(z) * np.sqrt(2.0 / np.pi))))
    return slope, p

def media_movel_centrada(serie, janela=10):
    return pd.Series(serie).rolling(window=janela, center=True, min_periods=1).mean().values

def mascara_rn(da, shp):
    """Cria máscara para o estado do Rio Grande do Norte"""
    gdf = gpd.read_file(shp)
    rn = gdf[gdf["sigla"] == "RN"].geometry.union_all()
    mask = np.zeros(da.shape, bool)
    for i, lat in enumerate(da.lat.values):
        for j, lon in enumerate(da.lon.values):
            mask[i, j] = rn.contains(Point(lon, lat))
    return mask

SHAPE_MUNICIPIOS = r"F:\Mestrado\Dados_INMET\shapefiles\lml_municipio_a.shp"
gdf_mun = gpd.read_file(SHAPE_MUNICIPIOS).to_crs("EPSG:4326")

def identificar_municipio(lat, lon, gdf_mun):
    p = Point(lon, lat)
    r = gdf_mun[gdf_mun.geometry.contains(p)]
    if r.empty:
        return "Município não identificado"
    return r.iloc[0]["nome"]

def processar_otimizado(filepath, col, tipo, lat="latitude", lon="longitude", date_col="data", chunk_size=5_000_000):
    print(f"  Lendo arquivo parquet em chunks (chunk_size={chunk_size:,})...")
    parquet_file = pq.ParquetFile(filepath)
    meses = {"DJF":[12,1,2],"MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11]}
    acumuladores = {e:[] for e in meses}
    total_rows = 0
    
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()
        total_rows += len(df)
        print(f"    Processando chunk... Total: {total_rows:,} linhas")
        df["time"] = pd.to_datetime(df[date_col])
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        
        for e, ml in meses.items():
            dfe = df[df["month"].isin(ml)]
            if len(dfe) == 0:
                continue
            if tipo == "precip":
                agg = dfe.groupby([lat, lon, "year"], observed=True)[col].sum().reset_index()
            else:
                agg = dfe.groupby([lat, lon, "year"], observed=True)[col].mean().reset_index()
            acumuladores[e].append(agg)
        del df
        gc.collect()
    
    print(f"  Total processado: {total_rows:,} linhas")
    print("  Consolidando e calculando tendências...")
    
    res = {}
    for e in meses:
        if not acumuladores[e]:
            continue
        dfc = pd.concat(acumuladores[e], ignore_index=True)
        if tipo == "precip":
            dfc = dfc.groupby([lat, lon, "year"], observed=True)[col].sum().reset_index()
        else:
            dfc = dfc.groupby([lat, lon, "year"], observed=True)[col].mean().reset_index()
        da = dfc.rename(columns={lat:"lat", lon:"lon", col:"v"}).set_index(["year","lat","lon"]).to_xarray()["v"]
        slope, p = xr.apply_ufunc(
            mk_sen_fast, da,
            input_core_dims=[["year"]],
            output_core_dims=[[],[]],
            vectorize=True,
            output_dtypes=[float,float]
        )
        res[e] = {"slope":slope,"p":p,"data":da}
        print(f"    {e}: slope = [{float(slope.min()):.4f}, {float(slope.max()):.4f}]")
        del dfc
        gc.collect()
    return res

def analise_integrada_ponto(res_dict, estacao, lat, lon, municipio, out, nome_arquivo=None):
    """Análise integrada de todas as variáveis climáticas em um mesmo ponto"""
    
    if nome_arquivo is None:
        nome_arquivo = municipio
    
    variaveis = {
        'temp': {'res': res_dict['temp'], 'nome': 'Temperatura', 'unidade': '°C', 'cor': CORES['temp']},
        'rad': {'res': res_dict['rad'], 'nome': 'Radiação', 'unidade': 'W/m²', 'cor': CORES['rad']},
        'prec': {'res': res_dict['prec'], 'nome': 'Precipitação', 'unidade': 'mm', 'cor': CORES['prec']}
    }
    
    dados = {}
    for var_key, var_info in variaveis.items():
        res = var_info['res']
        data = res[estacao]["data"]
        serie = data.sel(lat=lat, lon=lon, method="nearest").values
        anos = data.year.values
        slope_val, p_val = mk_sen_fast(serie)
        
        dados[var_key] = {
            'serie': serie, 'anos': anos, 'mm': media_movel_centrada(serie, 10),
            'clim': serie.mean(), 'anom': serie - serie.mean(),
            'slope': slope_val, 'p_value': p_val
        }
    
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(f'Análise Integrada de Variáveis Climáticas - {municipio} ({estacao})',
                 fontsize=24, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(3, 3, left=0.06, right=0.96, bottom=0.06, top=0.92, 
                          hspace=0.35, wspace=0.25)
    
    row = 0
    for var_key, var_info in variaveis.items():
        d = dados[var_key]
        nome, unidade, cor = var_info['nome'], var_info['unidade'], var_info['cor']
        anos, serie, mm, clim, anom = d['anos'], d['serie'], d['mm'], d['clim'], d['anom']
        slope, p_val = d['slope'], d['p_value']
        trend = slope * (anos - anos[0]) + serie[0]
        
        # Série Histórica
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.plot(anos, serie, 'o-', color=cor, linewidth=2.5, markersize=7, alpha=0.7, 
                markeredgecolor='white', markeredgewidth=1.5, label='Dados anuais')
        ax1.plot(anos, trend, '--', color='#2C3E50', linewidth=3, 
                label=f'Tendência: {slope:.3f} {unidade}/ano' + 
                      (f' (p={p_val:.3f})' if not np.isnan(p_val) else ''))
        ax1.set_xlabel('Ano', fontsize=16, fontweight='bold')
        ax1.set_ylabel(unidade, fontsize=16, fontweight='bold')
        ax1.set_title(f'{nome}\nSérie Histórica (1990-2024)', fontsize=17, fontweight='bold', pad=12)
        ax1.legend(loc='best', fontsize=13, framealpha=0.95, edgecolor='gray')
        ax1.grid(True, alpha=0.3, linestyle='--', color=CORES["grid"])
        ax1.set_facecolor('#f8f9fa')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        if p_val < 0.05:
            ax1.text(0.02, 0.98, '★ Significativo (p<0.05)', transform=ax1.transAxes, 
                    fontsize=12, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        # Série Filtrada
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.plot(anos, mm, color=cor, linewidth=4, label='Média móvel (10 anos)', alpha=0.9)
        ax2.fill_between(anos, mm, clim, where=(mm >= clim), alpha=0.25, 
                        color=CORES["positivo"], label='Acima da climatologia')
        ax2.fill_between(anos, mm, clim, where=(mm < clim), alpha=0.25, 
                        color=CORES["negativo"], label='Abaixo da climatologia')
        ax2.axhline(clim, color='black', linestyle=':', linewidth=2, alpha=0.7, 
                   label=f'Climatologia: {clim:.2f} {unidade}')
        ax2.set_xlabel('Ano', fontsize=16, fontweight='bold')
        ax2.set_ylabel(unidade, fontsize=16, fontweight='bold')
        ax2.set_title(f'{nome}\nSérie Filtrada (10 anos)', fontsize=17, fontweight='bold', pad=12)
        ax2.legend(loc='best', fontsize=12, framealpha=0.95, edgecolor='gray')
        ax2.grid(True, alpha=0.3, linestyle='--', color=CORES["grid"])
        ax2.set_facecolor('#f8f9fa')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Anomalias
        ax3 = fig.add_subplot(gs[row, 2])
        colors = [CORES["positivo"] if a > 0 else CORES["negativo"] for a in anom]
        ax3.bar(anos, anom, color=colors, width=0.85, alpha=0.85, edgecolor='white', linewidth=0.8)
        ax3.axhline(0, color='black', linewidth=2.5, zorder=3)
        ax3.set_xlabel('Ano', fontsize=16, fontweight='bold')
        ax3.set_ylabel(f'Anomalia ({unidade})', fontsize=16, fontweight='bold')
        ax3.set_title(f'{nome}\nAnomalias', fontsize=17, fontweight='bold', pad=12)
        ax3.grid(True, alpha=0.3, linestyle='--', color=CORES["grid"], axis='y')
        ax3.set_facecolor('#f8f9fa')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        legend_elements = [
            mpatches.Patch(facecolor=CORES["positivo"], alpha=0.8, edgecolor='white', 
                          linewidth=1.5, label='Anomalia positiva'),
            mpatches.Patch(facecolor=CORES["negativo"], alpha=0.8, edgecolor='white', 
                          linewidth=1.5, label='Anomalia negativa')
        ]
        ax3.legend(handles=legend_elements, loc='best', fontsize=12, framealpha=0.95, edgecolor='gray')
        stats_text = f'Máx: {anom.max():.2f}\nMín: {anom.min():.2f}\nDesvio: {anom.std():.2f}'
        ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes, fontsize=11, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        row += 1
    
    plt.savefig(f"{out}/Analise_Integrada_{nome_arquivo.replace(' ', '_')}_{estacao}.png", 
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Análise integrada salva: {municipio} ({estacao})")
    
    return {var: {'slope': dados[var]['slope'], 'p_value': dados[var]['p_value']} 
            for var in variaveis.keys()}

def selecionar_pontos_geograficos_rn(res_temp, shp):
    """
    Seleciona 10 pontos estrategicamente distribuídos pelo Rio Grande do Norte
    """
    pontos_alvo = [
        (-4.95, -37.35, "Noroeste"),     # Extremo noroeste (Apodi)
        (-5.20, -36.50, "Norte"),        # Norte (Touros)
        (-5.10, -35.20, "Nordeste"),     # Extremo nordeste (Costa)
        (-5.80, -38.00, "Oeste"),        # Extremo oeste (Pau dos Ferros)
        (-5.80, -36.50, "Centro"),       # Centro do estado
        (-5.80, -35.20, "Capital"),      # Natal (capital)
        (-6.40, -37.80, "Sudoeste"),     # Sudoeste (Caicó)
        (-6.20, -36.50, "Centro-Sul"),   # Centro-sul
        (-6.60, -35.80, "Sul"),          # Sul (Serra Negra do Norte)
        (-6.00, -35.00, "Leste"),        # Leste (litoral sul)
    ]
    
    slope = res_temp["DJF"]["slope"]
    mask_rn = mascara_rn(slope, shp)
    
    pontos_selecionados = []
    
    # Cria grades de lat/lon
    lat_grid, lon_grid = np.meshgrid(slope.lat.values, slope.lon.values, indexing='ij')
    
    for lat_alvo, lon_alvo, regiao in pontos_alvo:
        # Calcula distâncias em grade 2D
        distancias = np.sqrt((lat_grid - lat_alvo)**2 + (lon_grid - lon_alvo)**2)
        
        # Cria máscara de pontos válidos no RN
        valid_mask = mask_rn & ~np.isnan(slope.values)
        
        if not np.any(valid_mask):
            continue
            
        # Encontra ponto mais próximo válido
        distancias_masked = np.where(valid_mask, distancias, np.inf)
        idx_min = np.unravel_index(np.argmin(distancias_masked), distancias.shape)
        
        lat_sel = float(slope.lat[idx_min[0]])
        lon_sel = float(slope.lon[idx_min[1]])
        
        pontos_selecionados.append((lat_sel, lon_sel, regiao))
    
    return pontos_selecionados


if __name__ == "__main__":
    print("="*80)
    print(" " * 17 + "ANÁLISE INTEGRADA - RIO GRANDE DO NORTE")
    print(" " * 15 + "10 Pontos Distribuídos Geograficamente")
    print("="*80)
    
    temp_path = r"F:\rn\ERA5_TemperaturaMedia_RN_1990_2024.parquet"
    rad_path = r"F:\rn\SIS_RN.parquet"
    prec_path = r"F:\rn\CHIRPS_RN_diaria_por_pixel_1990_2024.parquet"
    shp = r"F:\Mestrado\Dados_INMET\shapefiles\lml_unidade_federacao_a.shp"
    out = r"F:\resultados_RN_10pontos"
    os.makedirs(out, exist_ok=True)
    
    print("\n[1/3] Processando Temperatura...")
    res_t = processar_otimizado(temp_path, "temp_celsius", "media", chunk_size=5_000_000)
    
    print("\n[2/3] Processando Radiação...")
    res_r = processar_otimizado(rad_path, "SIS", "media", lat="lat", lon="lon", 
                                date_col="time", chunk_size=5_000_000)
    
    print("\n[3/3] Processando Precipitação...")
    res_p = processar_otimizado(prec_path, "precipitation", "precip", chunk_size=10_000_000)
    
    res_dict = {'temp': res_t, 'rad': res_r, 'prec': res_p}
    
    print("\n" + "="*80)
    print("SELECIONANDO 10 PONTOS DISTRIBUÍDOS PELO RIO GRANDE DO NORTE")
    print("="*80)
    
    pontos = selecionar_pontos_geograficos_rn(res_t, shp)
    
    print(f"\n✓ {len(pontos)} pontos selecionados:")
    for i, (lat, lon, regiao) in enumerate(pontos, 1):
        municipio = identificar_municipio(lat, lon, gdf_mun)
        print(f"  {i:2d}. {regiao:12s} - {municipio:30s} (Lat: {lat:6.2f}°, Lon: {lon:7.2f}°)")
    
    print("\n" + "="*80)
    print("GERANDO ANÁLISES INTEGRADAS")
    print("="*80)
    
    total_analises = 0
    for estacao in ["DJF", "MAM", "JJA", "SON"]:
        print(f"\n{'='*80}")
        print(f"Estação: {estacao}")
        print('='*80)
        
        for i, (lat, lon, regiao) in enumerate(pontos, 1):
            municipio = identificar_municipio(lat, lon, gdf_mun)
            print(f"\n  [{i}/10] {regiao} - {municipio}")
            
            stats = analise_integrada_ponto(res_dict, estacao, lat, lon, 
                                           municipio, out, 
                                           nome_arquivo=f"{i:02d}_{regiao}_{municipio}")
            
            print(f"     • Temperatura:   {stats['temp']['slope']:>7.4f} °C/ano   (p={stats['temp']['p_value']:.3f})")
            print(f"     • Radiação:      {stats['rad']['slope']:>7.4f} W/m²/ano (p={stats['rad']['p_value']:.3f})")
            print(f"     • Precipitação:  {stats['prec']['slope']:>7.2f} mm/ano   (p={stats['prec']['p_value']:.3f})")
            
            total_analises += 1
    
    print("\n" + "="*80)
    print("✓ PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*80)
    print(f"✓ Total de análises: {total_analises} gráficos (10 pontos × 4 estações)")
    print(f"✓ Arquivos salvos em: {out}")
    print("="*80)