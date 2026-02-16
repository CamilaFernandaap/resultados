import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from numba import njit
import pyarrow.parquet as pq
import gc
import os
from shapely.geometry import Point

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "figure.titlesize": 26
})

@njit(fastmath=True)
def mk_sen_fast(v):
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

def processar_sazonal_decadal(filepath, col, tipo, lat="latitude", lon="longitude", date_col="data", chunk_size=5_000_000):
    """
    Processa dados e calcula tendências por estação do ano, expressas em unidades DECADAIS.
    Estações: DJF, MAM, JJA, SON
    Tendências calculadas para todo o período (1990-2024) mas expressas por década.
    """
    parquet_file = pq.ParquetFile(filepath)
    meses = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
    acumuladores = {e: [] for e in meses}
    
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()
        df["time"] = pd.to_datetime(df[date_col])
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        
        for e, ml in meses.items():
            dfe = df[df["month"].isin(ml)]
            if len(dfe) == 0:
                continue
            
            if tipo == "precip":
                # Para precipitação: soma mensal primeiro
                agg = dfe.groupby([lat, lon, "year", "month"], observed=True)[col].sum().reset_index()
                # Depois média anual (média das somas mensais)
                agg = agg.groupby([lat, lon, "year"], observed=True)[col].mean().reset_index()
            else:
                # Para temperatura e radiação: média direta
                agg = dfe.groupby([lat, lon, "year"], observed=True)[col].mean().reset_index()
            
            acumuladores[e].append(agg)
        
        del df
        gc.collect()
    
    res = {}
    
    for e in meses:
        if not acumuladores[e]:
            continue
        
        dfc = pd.concat(acumuladores[e], ignore_index=True)
        
        if tipo == "precip":
            dfc = dfc.groupby([lat, lon, "year"], observed=True)[col].mean().reset_index()
        else:
            dfc = dfc.groupby([lat, lon, "year"], observed=True)[col].mean().reset_index()
        
        # Converte para xarray
        da = dfc.rename(columns={lat: "lat", lon: "lon", col: "v"}).set_index(["year", "lat", "lon"]).to_xarray()["v"]
        
        # Calcula Mann-Kendall e Sen's slope (em unidades por ANO)
        slope_anual, p = xr.apply_ufunc(
            mk_sen_fast,
            da,
            input_core_dims=[["year"]],
            output_core_dims=[[], []],
            vectorize=True,
            output_dtypes=[float, float]
        )
        
        # Converte slope de "por ano" para "por década" (multiplica por 10)
        slope_decadal = slope_anual * 10.0
        
        res[e] = {"slope": slope_decadal, "p": p}
        
        del dfc
        gc.collect()
    
    return res

def mascara_estado(da, geom):
    lats = da.lat.values
    lons = da.lon.values
    mask = np.zeros((len(lats), len(lons)), dtype=bool)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if geom.contains(Point(lon, lat)):
                mask[i, j] = True
    return da.where(mask)

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from shapely.geometry import Polygon

def plot_mapas_tendencia(res, shp_uf, sigla, titulo, unidade, cmap, limites_por_estacao, out_png):
    """
    Plota mapas de tendência sazonal em unidades decadais.
    Layout 2x2 para DJF, MAM, JJA, SON.
    """
    gdf = gpd.read_file(shp_uf).to_crs("EPSG:4326")
    geom = gdf[gdf["sigla"] == sigla].geometry.union_all()

    xmin, ymin, xmax, ymax = geom.bounds
    dx = (xmax - xmin) * 0.03
    dy = (ymax - ymin) * 0.03
    xmin -= dx; xmax += dx
    ymin -= dy; ymax += dy

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    axes = axes.flatten()
    mappables = []

    for ax, estacao in zip(axes, ["DJF", "MAM", "JJA", "SON"]):
        slope = mascara_estado(res[estacao]["slope"], geom)
        pval = mascara_estado(res[estacao]["p"], geom)

        vmin, vmax = limites_por_estacao[estacao]
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        # Mapa principal (tendência)
        im = ax.pcolormesh(
            slope.lon,
            slope.lat,
            slope.values,
            cmap=cmap,
            norm=norm,
            shading="auto"
        )
        mappables.append(im)

        # Máscara de significância (p <= 0.05)
        sig_mask = (pval.values <= 0.05) & (~np.isnan(slope.values))

        # Converte cada célula significativa em polígono
        polys = []
        for i in range(len(slope.lat)-1):
            for j in range(len(slope.lon)-1):
                if sig_mask[i, j]:
                    polys.append(Polygon([
                        (slope.lon[j], slope.lat[i]),
                        (slope.lon[j+1], slope.lat[i]),
                        (slope.lon[j+1], slope.lat[i+1]),
                        (slope.lon[j], slope.lat[i+1])
                    ]))

        if polys:
            gpd.GeoSeries(polys, crs="EPSG:4326").plot(
                ax=ax,
                facecolor="none",
                edgecolor="black",
                hatch="////",
                linewidth=0
            )

        # Contorno do estado
        gpd.GeoSeries(geom, crs="EPSG:4326").boundary.plot(ax=ax, linewidth=1.8, color="black")

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel("Longitude", fontsize=18)
        ax.set_ylabel("Latitude", fontsize=18)
        ax.set_title(estacao, fontsize=22, fontweight='bold', pad=10)

    # Barra de cores compartilhada
    cbar = fig.colorbar(mappables[0], ax=axes, orientation="vertical", shrink=0.9)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(unidade, fontsize=22)

    fig.suptitle(titulo, fontsize=26, fontweight='bold', y=0.995)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Mapa salvo: {out_png}")

if __name__ == "__main__":
    print("="*70)
    print("ANÁLISE DE TENDÊNCIAS SAZONAIS EM TERMOS DECADAIS")
    print("Mann-Kendall & Sen's Slope - 1990-2024")
    print("Tendências expressas por DÉCADA (10 anos)")
    print("="*70)
    
    shp_uf = r"F:\Mestrado\Dados_INMET\shapefiles\lml_unidade_federacao_a.shp"
    out_dir = r"F:\resultados_mapas_tendencia_sazonal_decadal"
    os.makedirs(out_dir, exist_ok=True)

    prec_path = r"F:\rn\CHIRPS_RN_diaria_por_pixel_1990_2024.parquet"
    rad_path = r"F:\rn\SIS_RN.parquet"
    temp_path = r"F:\rn\ERA5_TemperaturaMedia_RN_1990_2024.parquet"

    # Processa as tendências sazonais (em termos decadais)
    print("\n[1/3] Processando Precipitação...")
    res_p = processar_sazonal_decadal(prec_path, "precipitation", "precip")
    
    print("\n[2/3] Processando Radiação...")
    res_r = processar_sazonal_decadal(rad_path, "SIS", "media", lat="lat", lon="lon", date_col="time")
    
    print("\n[3/3] Processando Temperatura...")
    res_t = processar_sazonal_decadal(temp_path, "temp_celsius", "media")

    # Define limites da escala de cores por estação
    # Valores agora são em unidades POR DÉCADA (10x maiores que por ano)
    limites_prec = {
        "DJF": (-30, 30),   # mm/década
        "MAM": (-30, 30),
        "JJA": (-30, 30),
        "SON": (-30, 30)
    }

    limites_rad = {
        "DJF": (-15, 15),   # W/m²/década
        "MAM": (-15, 15),
        "JJA": (-15, 15),
        "SON": (-15, 15)
    }

    limites_temp = {
        "DJF": (-0.4, 0.4),  # °C/década
        "MAM": (-0.4, 0.4),
        "JJA": (-0.4, 0.4),
        "SON": (-0.4, 0.4)
    }

    print("\n[4/4] Gerando Mapas de Tendências Sazonais (expressas por década)...")
    
    plot_mapas_tendencia(
        res_p,
        shp_uf,
        "RN",
        "Tendências Sazonais de Precipitação – Rio Grande do Norte (1990–2020)",
        "mm/década",
        "BrBG",
        limites_prec,
        os.path.join(out_dir, "prec_tendencia_sazonal_decadal_rn.png")
    )

    plot_mapas_tendencia(
        res_r,
        shp_uf,
        "RN",
        "Tendências Sazonais de Radiação – Rio Grande do Norte (1990–2020)",
        "W/m²/década",
        "YlOrBr",
        limites_rad,
        os.path.join(out_dir, "rad_tendencia_sazonal_decadal_rn.png")
    )

    plot_mapas_tendencia(
        res_t,
        shp_uf,
        "RN",
        "Tendências Sazonais de Temperatura – Rio Grande do Norte (1990–2020)",
        "°C/década",
        "RdBu_r",
        limites_temp,
        os.path.join(out_dir, "temp_tendencia_sazonal_decadal_rn.png")
    )

    print("\n" + "="*70)
    print(" PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print(f" Mapas gerados em: {out_dir}")
    print("  - Tendências sazonais (DJF, MAM, JJA, SON)")
    print("  - Expressas em unidades por DÉCADA (mm/década, °C/década, W/m²/década)")
    print("  - Período: 1990-2024 (34 anos)")
    print("="*70)