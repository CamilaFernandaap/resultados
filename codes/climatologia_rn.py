import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, BoundaryNorm
from shapely.geometry import Point
import os
import pyarrow.parquet as pq
import gc

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5)  # sobe tudo de forma consistente

plt.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    "figure.titlesize": 26,
    "axes.facecolor": "#f8f9fa",
    "figure.facecolor": "white",
    "grid.alpha": 0.3,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2
})


def criar_cmap_precipitacao():
    return LinearSegmentedColormap.from_list(
        "prec",
        ['#FFFFFF','#E0F3F8','#ABD9E9','#74ADD1','#4575B4',
         '#313695','#006837','#1A9850','#66BD63','#FEE08B','#FDAE61'],
        N=256
    )

def criar_cmap_temperatura():
    return LinearSegmentedColormap.from_list(
        "temp",
        ['#313695','#4575B4','#74ADD1','#ABD9E9','#E0F3F8',
         '#FFFFBF','#FEE090','#FDAE61','#F46D43','#D73027','#A50026'],
        N=256
    )

def criar_cmap_radiacao():
    return LinearSegmentedColormap.from_list(
        "rad",
        ['#FFFFD4','#FED98E','#FE9929','#D95F0E','#993404'],
        N=256
    )

def processar_climatologia(filepath, col, tipo, lat="latitude", lon="longitude", 
                           date_col="data", chunk_size=5_000_000):
    """
    Calcula a normal climatológica (média 1990-2024) para cada estação do ano
    """
    print(f"  Lendo arquivo parquet em chunks (chunk_size={chunk_size:,})...")
    parquet_file = pq.ParquetFile(filepath)
    meses = {"DJF":[12,1,2], "MAM":[3,4,5], "JJA":[6,7,8], "SON":[9,10,11]}
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
    print("  Consolidando e calculando climatologia...")
    
    climatologias = {}
    for e in meses:
        if not acumuladores[e]:
            continue
        dfc = pd.concat(acumuladores[e], ignore_index=True)
        if tipo == "precip":
            dfc = dfc.groupby([lat, lon, "year"], observed=True)[col].sum().reset_index()
        else:
            dfc = dfc.groupby([lat, lon, "year"], observed=True)[col].mean().reset_index()
        
        # Cria xarray
        da = dfc.rename(columns={lat:"lat", lon:"lon", col:"v"}).set_index(["year","lat","lon"]).to_xarray()["v"]
        
        # Calcula estatísticas climatológicas
        clim_media = da.mean("year")
        clim_std = da.std("year")
        clim_min = da.min("year")
        clim_max = da.max("year")
        clim_p25 = da.quantile(0.25, "year")
        clim_p75 = da.quantile(0.75, "year")
        
        climatologias[e] = {
            "media": clim_media,
            "std": clim_std,
            "min": clim_min,
            "max": clim_max,
            "p25": clim_p25,
            "p75": clim_p75,
            "data": da  # Dados completos para análises adicionais
        }
        
        print(f"    {e}: média = {float(clim_media.mean()):.2f}, "
              f"std = {float(clim_std.mean()):.2f}")
        
        del dfc
        gc.collect()
    
    return climatologias

def plot_climatologia_mapas(clim_dict, titulo, unidade, cmap, shp, out):
    """
    Gera mapas da normal climatológica para as 4 estações
    """
    gdf = gpd.read_file(shp)
    rn = gdf[gdf["sigla"]=="RN"]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    for ax, e in zip(axes.flat, ["DJF", "MAM", "JJA", "SON"]):
        clim_media = clim_dict[e]["media"]
        
        # Plota a média climatológica
        im = clim_media.plot(ax=ax, cmap=cmap, add_colorbar=False)
        rn.boundary.plot(ax=ax, color="black", linewidth=2.5)
        
        ax.set_title(f"{e}", fontsize=24)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(-38.6, -34.8)
        ax.set_ylim(-7.1, -4.8)
        ax.set_xlabel('Longitude', fontsize=18)
        ax.set_ylabel('Latitude', fontsize=18)
    
    # Colorbar compartilhada
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax).set_label(unidade, fontsize=22)
    
    fig.suptitle(f"Normal Climatológica (1990-2024) - {titulo} - Rio Grande do Norte", 
                 fontsize=24, y=0.96)
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.08, top=0.92, hspace=0.25, wspace=0.25)
    
    plt.savefig(f"{out}/Climatologia_{titulo}_RN_mapas.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Salvo: Climatologia_{titulo}_RN_mapas.png")

def plot_climatologia_estatisticas(clim_dict, titulo, unidade, shp, out):
    """
    Gera mapas com estatísticas: média, desvio padrão, amplitude
    """
    gdf = gpd.read_file(shp)
    rn = gdf[gdf["sigla"]=="RN"]
    
    # Escolhe uma estação para visualizar (DJF como exemplo)
    estacao = "DJF"
    clim = clim_dict[estacao]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Painel 1: Média
    ax1 = axes[0, 0]
    im1 = clim["media"].plot(ax=ax1, cmap=criar_cmap_temperatura(), add_colorbar=True, 
                             cbar_kwargs={'label': unidade})
    rn.boundary.plot(ax=ax1, color="black", linewidth=2.5)
    ax1.set_title(f"Média Climatológica", fontsize=18, fontweight='bold')
    ax1.set_xlim(-38.5, -34.8)
    ax1.set_ylim(-6.9, -4.8)
    
    # Painel 2: Desvio Padrão
    ax2 = axes[0, 1]
    im2 = clim["std"].plot(ax=ax2, cmap='YlOrRd', add_colorbar=True,
                          cbar_kwargs={'label': unidade})
    rn.boundary.plot(ax=ax2, color="black", linewidth=2.5)
    ax2.set_title(f"Desvio Padrão", fontsize=18, fontweight='bold')
    ax2.set_xlim(-38.6, -34.8)
    ax2.set_ylim(-7.1, -4.8)
    
    # Painel 3: Mínimo
    ax3 = axes[1, 0]
    im3 = clim["min"].plot(ax=ax3, cmap='Blues', add_colorbar=True,
                          cbar_kwargs={'label': unidade})
    rn.boundary.plot(ax=ax3, color="black", linewidth=2.5)
    ax3.set_title(f"Mínimo (1990-2024)", fontsize=18, fontweight='bold')
    ax3.set_xlim(-38.6, -34.8)
    ax3.set_ylim(-7.1, -4.8)
    
    # Painel 4: Máximo
    ax4 = axes[1, 1]
    im4 = clim["max"].plot(ax=ax4, cmap='Reds', add_colorbar=True,
                          cbar_kwargs={'label': unidade})
    rn.boundary.plot(ax=ax4, color="black", linewidth=2.5)
    ax4.set_title(f"Máximo (1990-2024)", fontsize=18, fontweight='bold')
    ax4.set_xlim(-38.6, -34.8)
    ax4.set_ylim(-7.1, -4.8)
    
    fig.suptitle(f"Estatísticas Climatológicas - {titulo} ({estacao}) - Rio Grande do Norte", 
                 fontsize=24, y=0.96)
    fig.subplots_adjust(left=0.05, right=0.96, bottom=0.08, top=0.92, hspace=0.3, wspace=0.3)
    
    plt.savefig(f"{out}/Climatologia_{titulo}_RN_estatisticas_{estacao}.png", 
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Salvo: Climatologia_{titulo}_RN_estatisticas_{estacao}.png")

def plot_climatologia_boxplots(clim_dict, titulo, unidade, out):
    """
    Cria boxplots comparativos entre estações mostrando 
    a distribuição completa dos valores
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    dados_box = []
    labels = []
    
    for estacao in ["DJF", "MAM", "JJA", "SON"]:
        # Achata os dados espaciais para ter todos os valores
        valores = clim_dict[estacao]["data"].values.flatten()
        # Remove NaNs
        valores = valores[~np.isnan(valores)]
        dados_box.append(valores)
        labels.append(estacao)
    
    bp = ax.boxplot(dados_box, labels=labels, 
                     patch_artist=True,
                     widths=0.6,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    ax.set_ylabel(unidade, fontsize=22)
    ax.set_xlabel("Estação do Ano", fontsize=22)
    ax.set_title(f"Distribuição Estatística por Estação – {titulo} (1990–2024) – Rio Grande do Norte",
                 fontsize=24, fontweight="bold")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(f"{out}/Climatologia_{titulo}_RN_boxplot.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    
def exportar_estatisticas_csv(clim_temp, clim_rad, clim_prec, out):
    dados = []
    
    for estacao in ["DJF", "MAM", "JJA", "SON"]:
        # Temperatura
        dados.append({
            'Estacao': estacao,
            'Variavel': 'Temperatura',
            'Media': float(clim_temp[estacao]["media"].mean()),
            'Desvio_Padrao': float(clim_temp[estacao]["std"].mean()),
            'Minimo': float(clim_temp[estacao]["min"].min()),
            'Maximo': float(clim_temp[estacao]["max"].max()),
            'P25': float(clim_temp[estacao]["p25"].mean()),
            'P75': float(clim_temp[estacao]["p75"].mean()),
            'Unidade': '°C'
        })
        
        # Radiação
        dados.append({
            'Estacao': estacao,
            'Variavel': 'Radiacao',
            'Media': float(clim_rad[estacao]["media"].mean()),
            'Desvio_Padrao': float(clim_rad[estacao]["std"].mean()),
            'Minimo': float(clim_rad[estacao]["min"].min()),
            'Maximo': float(clim_rad[estacao]["max"].max()),
            'P25': float(clim_rad[estacao]["p25"].mean()),
            'P75': float(clim_rad[estacao]["p75"].mean()),
            'Unidade': 'W/m²'
        })
        
        # Precipitação
        dados.append({
            'Estacao': estacao,
            'Variavel': 'Precipitacao',
            'Media': float(clim_prec[estacao]["media"].mean()),
            'Desvio_Padrao': float(clim_prec[estacao]["std"].mean()),
            'Minimo': float(clim_prec[estacao]["min"].min()),
            'Maximo': float(clim_prec[estacao]["max"].max()),
            'P25': float(clim_prec[estacao]["p25"].mean()),
            'P75': float(clim_prec[estacao]["p75"].mean()),
            'Unidade': 'mm'
        })
    
    df = pd.DataFrame(dados)
    df.to_csv(f"{out}/Climatologia_RN_estatisticas.csv", index=False)
    print(f"   Salvo: Climatologia_RN_estatisticas.csv")
    
    return df


if __name__ == "__main__":
    print("="*80)
    print(" "*15 + "NORMAL CLIMATOLÓGICA - RIO GRANDE DO NORTE")
    print(" "*25 + "(Período: 1990-2024)")
    print("="*80)
    
    temp_path = r"F:\rn\ERA5_TemperaturaMedia_RN_1990_2024.parquet"
    rad_path = r"F:\rn\SIS_RN.parquet"
    prec_path = r"F:\rn\CHIRPS_RN_diaria_por_pixel_1990_2024.parquet"
    shp = r"F:\Mestrado\Dados_INMET\shapefiles\lml_unidade_federacao_a.shp"
    out = r"F:\resultados_RN_climatologia"
    os.makedirs(out, exist_ok=True)
    
    print("\n[1/3] Calculando climatologia de Temperatura...")
    clim_temp = processar_climatologia(temp_path, "temp_celsius", "media", chunk_size=5_000_000)
    
    print("\n[2/3] Calculando climatologia de Radiação...")
    clim_rad = processar_climatologia(rad_path, "SIS", "media", lat="lat", lon="lon", 
                                      date_col="time", chunk_size=5_000_000)
    
    print("\n[3/3] Calculando climatologia de Precipitação...")
    clim_prec = processar_climatologia(prec_path, "precipitation", "precip", chunk_size=10_000_000)
    
    # Gera visualizações
    print("\n" + "="*80)
    print("GERANDO MAPAS CLIMATOLÓGICOS")
    print("="*80)
    
    print("\n[1/4] Mapas de Temperatura...")
    plot_climatologia_mapas(clim_temp, "Temperatura", "°C", criar_cmap_temperatura(), shp, out)
    plot_climatologia_boxplots(clim_temp, "Temperatura", "°C", out)  # <-- sem shp
    plot_climatologia_estatisticas(clim_temp, "Temperatura", "°C", shp, out)
    
    print("\n[2/4] Mapas de Radiação...")
    plot_climatologia_mapas(clim_rad, "Radiação", "W/m²", criar_cmap_radiacao(), shp, out)
    plot_climatologia_boxplots(clim_rad, "Radiação", "W/m²", out)  # <-- sem shp
    plot_climatologia_estatisticas(clim_rad, "Radiação", "W/m²", shp, out)
    
    print("\n[3/4] Mapas de Precipitação...")
    plot_climatologia_mapas(clim_prec, "Precipitação", "mm", criar_cmap_precipitacao(), shp, out)
    plot_climatologia_boxplots(clim_prec, "Precipitação", "mm", out)  # <-- sem shp
    plot_climatologia_estatisticas(clim_prec, "Precipitação", "mm", shp, out)
    
    
    print("\n[4/4] Exportando estatísticas...")
    df_stats = exportar_estatisticas_csv(clim_temp, clim_rad, clim_prec, out)
    
    print("\n" + "="*80)
    print("RESUMO DAS NORMAIS CLIMATOLÓGICAS")
    print("="*80)
    print("\n" + df_stats.to_string(index=False))
    
    print("\n" + "="*80)
    print(" PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*80)
    print(f" Arquivos gerados:")
    print(f"   • 3 mapas de médias climatológicas (4 estações cada)")
    print(f"   • 3 mapas de estatísticas detalhadas")
    print(f"   • 1 arquivo CSV com resumo estatístico")
    print(f"\n Diretório: {out}")
    print("="*80)