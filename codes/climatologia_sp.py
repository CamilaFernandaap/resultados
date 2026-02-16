import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import os
import pyarrow.parquet as pq
import gc

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5)

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
    parquet_file = pq.ParquetFile(filepath)
    meses = {"DJF":[12,1,2], "MAM":[3,4,5], "JJA":[6,7,8], "SON":[9,10,11]}
    acumuladores = {e:[] for e in meses}

    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()
        df["time"] = pd.to_datetime(df[date_col])
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month

        for e, ml in meses.items():
            dfe = df[df["month"].isin(ml)]
            if dfe.empty:
                continue
            if tipo == "precip":
                agg = dfe.groupby([lat, lon, "year"], observed=True)[col].sum().reset_index()
            else:
                agg = dfe.groupby([lat, lon, "year"], observed=True)[col].mean().reset_index()
            acumuladores[e].append(agg)

        del df
        gc.collect()

    climatologias = {}
    for e in meses:
        if not acumuladores[e]:
            continue
        dfc = pd.concat(acumuladores[e], ignore_index=True)
        if tipo == "precip":
            dfc = dfc.groupby([lat, lon, "year"], observed=True)[col].sum().reset_index()
        else:
            dfc = dfc.groupby([lat, lon, "year"], observed=True)[col].mean().reset_index()

        da = dfc.rename(columns={lat:"lat", lon:"lon", col:"v"}).set_index(
            ["year","lat","lon"]
        ).to_xarray()["v"]

        climatologias[e] = {
            "media": da.mean("year"),
            "std": da.std("year"),
            "min": da.min("year"),
            "max": da.max("year"),
            "p25": da.quantile(0.25, "year"),
            "p75": da.quantile(0.75, "year"),
            "data": da
        }

        del dfc
        gc.collect()

    return climatologias

def plot_climatologia_mapas(clim_dict, titulo, unidade, cmap, shp, out):
    gdf = gpd.read_file(shp)
    sp = gdf[gdf["sigla"]=="SP"]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    for ax, e in zip(axes.flat, ["DJF", "MAM", "JJA", "SON"]):
        im = clim_dict[e]["media"].plot(ax=ax, cmap=cmap, add_colorbar=False)
        sp.boundary.plot(ax=ax, color="black", linewidth=2.5)

        ax.set_title(e, fontsize=24, fontweight="bold")
        ax.set_xlim(-53.5, -44.0)
        ax.set_ylim(-25.5, -19.5)
        ax.set_xlabel("Longitude", fontsize=20)
        ax.set_ylabel("Latitude", fontsize=20)
        ax.tick_params(axis="both", labelsize=20)

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(unidade, fontsize=22)
    cbar.ax.tick_params(labelsize=20)

    fig.suptitle(
        f"Normal Climatológica (1990–2024) – {titulo} – São Paulo",
        fontsize=26, fontweight="bold", y=0.96
    )

    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.08, top=0.92,
                        hspace=0.25, wspace=0.25)

    plt.savefig(f"{out}/Climatologia_{titulo}_SP_mapas.png",
                dpi=300, bbox_inches="tight")
    plt.close()

def plot_climatologia_estatisticas(clim_dict, titulo, unidade, shp, out):
    gdf = gpd.read_file(shp)
    sp = gdf[gdf["sigla"]=="SP"]

    estacao = "DJF"
    clim = clim_dict[estacao]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    dados = [
        (clim["media"], criar_cmap_temperatura(), "Média Climatológica"),
        (clim["std"], "YlOrRd", "Desvio Padrão"),
        (clim["min"], "Blues", "Mínimo (1990–2024)"),
        (clim["max"], "Reds", "Máximo (1990–2024)")
    ]

    for ax, (campo, cmap, titulo_ax) in zip(axes.flat, dados):
        im = campo.plot(ax=ax, cmap=cmap, add_colorbar=True,
                         cbar_kwargs={"label": unidade})
        im.colorbar.ax.tick_params(labelsize=20)
        im.colorbar.set_label(unidade, fontsize=22)

        sp.boundary.plot(ax=ax, color="black", linewidth=2.5)
        ax.set_title(titulo_ax, fontsize=22, fontweight="bold")
        ax.set_xlim(-53.5, -44.0)
        ax.set_ylim(-25.5, -19.5)
        ax.tick_params(axis="both", labelsize=20)

    fig.suptitle(
        f"Estatísticas Climatológicas – {titulo} ({estacao}) – São Paulo",
        fontsize=26, fontweight="bold", y=0.96
    )

    fig.subplots_adjust(left=0.05, right=0.96, bottom=0.08, top=0.92,
                        hspace=0.3, wspace=0.3)

    plt.savefig(f"{out}/Climatologia_{titulo}_SP_estatisticas_{estacao}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

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
    ax.set_title(f"Distribuição Estatística por Estação – {titulo} (1990–2024) – São Paulo",
                 fontsize=24, fontweight="bold")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(f"{out}/Climatologia_{titulo}_SP_boxplot.png",
                dpi=300, bbox_inches="tight")
    plt.close()

def exportar_estatisticas_csv(clim_temp, clim_rad, clim_prec, out):
    dados = []

    for estacao in ["DJF", "MAM", "JJA", "SON"]:
        dados.append({
            "Estacao": estacao,
            "Variavel": "Temperatura",
            "Media": float(clim_temp[estacao]["media"].mean()),
            "Desvio_Padrao": float(clim_temp[estacao]["std"].mean()),
            "Minimo": float(clim_temp[estacao]["min"].min()),
            "Maximo": float(clim_temp[estacao]["max"].max()),
            "P25": float(clim_temp[estacao]["p25"].mean()),
            "P75": float(clim_temp[estacao]["p75"].mean()),
            "Unidade": "°C"
        })

        dados.append({
            "Estacao": estacao,
            "Variavel": "Radiacao",
            "Media": float(clim_rad[estacao]["media"].mean()),
            "Desvio_Padrao": float(clim_rad[estacao]["std"].mean()),
            "Minimo": float(clim_rad[estacao]["min"].min()),
            "Maximo": float(clim_rad[estacao]["max"].max()),
            "P25": float(clim_rad[estacao]["p25"].mean()),
            "P75": float(clim_rad[estacao]["p75"].mean()),
            "Unidade": "W/m²"
        })

        dados.append({
            "Estacao": estacao,
            "Variavel": "Precipitacao",
            "Media": float(clim_prec[estacao]["media"].mean()),
            "Desvio_Padrao": float(clim_prec[estacao]["std"].mean()),
            "Minimo": float(clim_prec[estacao]["min"].min()),
            "Maximo": float(clim_prec[estacao]["max"].max()),
            "P25": float(clim_prec[estacao]["p25"].mean()),
            "P75": float(clim_prec[estacao]["p75"].mean()),
            "Unidade": "mm"
        })

    df = pd.DataFrame(dados)
    df.to_csv(f"{out}/Climatologia_SP_estatisticas.csv", index=False)
    return df

if __name__ == "__main__":
    temp_path = r"F:\sp\ERA5_TemperaturaMedia_SP_1990_2024.parquet"
    rad_path = r"F:\sp\SIS_SP.parquet"
    prec_path = r"F:\sp\CHIRPS_SP_diaria_por_pixel_1990_2024.parquet"
    shp = r"F:\Mestrado\Dados_INMET\shapefiles\lml_unidade_federacao_a.shp"
    out = r"F:\resultados_SP_climatologia"

    os.makedirs(out, exist_ok=True)

    clim_temp = processar_climatologia(temp_path, "temp_celsius", "media")
    clim_rad = processar_climatologia(rad_path, "SIS", "media",
                                      lat="lat", lon="lon", date_col="time")
    clim_prec = processar_climatologia(prec_path, "precipitation", "precip")

    
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
    
    exportar_estatisticas_csv(clim_temp, clim_rad, clim_prec, out)
