import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
tif_1990 = r"C:\Users\camil\Downloads\mapbiomas\RN_1990_lclu_recortado.tif"
tif_2020 = r"C:\Users\camil\Downloads\mapbiomas\RN_2020_lclu_recortado.tif"

arquivo_temp = r"F:\rn\ERA5_TemperaturaMedia_RN_1990_2024.parquet"
arquivo_prec = r"F:\rn\CHIRPS_RN_diaria_por_pixel_1990_2024.parquet"
arquivo_rad = r"F:\rn\SIS_RN.parquet"

CLASSE_SAVANA = 4
TAMANHO_BLOCO = 100

def identificar_blocos_extremos(tif_1990, tif_2020, classe_alvo, tamanho_bloco=100):
    """Identifica blocos extremos"""
    print(f"\n{'='*80}")
    print(f"Identificando blocos extremos - Classe {classe_alvo}")
    print('='*80)
    
    with rasterio.open(tif_1990) as src:
        img_1990 = src.read(1)
        bounds = src.bounds
        transform = src.transform
        shape = img_1990.shape
        
        with rasterio.open(tif_2020) as src2020:
            img_2020 = src2020.read(1)
    
    print(f"  Shape: {shape}")
    
    mask_1990 = (img_1990 == classe_alvo).astype(float)
    mask_2020 = (img_2020 == classe_alvo).astype(float)
    mudanca = mask_2020 - mask_1990
    
    print(f"  Pixels PERDA: {np.sum(mudanca == -1):,}")
    print(f"  Pixels GANHO: {np.sum(mudanca == 1):,}")
    
    n_blocos_y = shape[0] // tamanho_bloco
    n_blocos_x = shape[1] // tamanho_bloco
    
    print(f"\n  Grade: {n_blocos_y} x {n_blocos_x} blocos")
    
    blocos_info = []
    
    for i in range(n_blocos_y):
        for j in range(n_blocos_x):
            y_start = i * tamanho_bloco
            y_end = (i + 1) * tamanho_bloco
            x_start = j * tamanho_bloco
            x_end = (j + 1) * tamanho_bloco
            
            bloco_mudanca = mudanca[y_start:y_end, x_start:x_end]
            
            total_pixels = bloco_mudanca.size
            pixels_perda = np.sum(bloco_mudanca == -1)
            pixels_ganho = np.sum(bloco_mudanca == 1)
            
            pct_perda = (pixels_perda / total_pixels) * 100
            pct_ganho = (pixels_ganho / total_pixels) * 100
            mudanca_liquida = pct_ganho - pct_perda
            
            # BBOX do bloco
            coords_cantos = []
            for y_idx in [y_start, y_end]:
                for x_idx in [x_start, x_end]:
                    lon, lat = rasterio.transform.xy(transform, y_idx, x_idx)
                    coords_cantos.append((lon, lat))
            
            lons = [c[0] for c in coords_cantos]
            lats = [c[1] for c in coords_cantos]
            
            buffer = 0.1
            
            blocos_info.append({
                'bloco_id': int(i * n_blocos_x + j),
                'y_start': y_start,
                'y_end': y_end,
                'x_start': x_start,
                'x_end': x_end,
                'bbox_minx': min(lons) - buffer,
                'bbox_maxx': max(lons) + buffer,
                'bbox_miny': min(lats) - buffer,
                'bbox_maxy': max(lats) + buffer,
                'pct_perda': pct_perda,
                'pct_ganho': pct_ganho,
                'mudanca_liquida': mudanca_liquida
            })
    
    df_blocos = pd.DataFrame(blocos_info)
    df_blocos_sorted = df_blocos.sort_values('mudanca_liquida')
    
    blocos_maior_perda = df_blocos_sorted.head(2).copy()
    blocos_maior_ganho = df_blocos_sorted.tail(2).copy()
    
    print(f"\n  {'='*76}")
    print(f"  BLOCOS COM MAIOR PERDA (top 2):")
    print(f"  {'='*76}")
    for idx, row in blocos_maior_perda.iterrows():
        print(f"    Bloco #{int(row['bloco_id']):4d} | Perda: {row['pct_perda']:5.1f}% | Ganho: {row['pct_ganho']:5.1f}%")
    
    print(f"\n  {'='*76}")
    print(f"  BLOCOS COM MAIOR GANHO (top 2):")
    print(f"  {'='*76}")
    for idx, row in blocos_maior_ganho.iterrows():
        print(f"    Bloco #{int(row['bloco_id']):4d} | Perda: {row['pct_perda']:5.1f}% | Ganho: {row['pct_ganho']:5.1f}%")
    
    return blocos_maior_perda, blocos_maior_ganho

def extrair_serie_temporal_bloco_individual(arquivo_parquet, bloco, variavel='temp'):
    """Extrai série temporal para UM bloco individual"""
    
    df = pd.read_parquet(arquivo_parquet)
    
    # Identificar colunas
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    date_col = next((c for c in df.columns if any(x in c.lower() for x in ['date', 'data', 'time'])), None)
    
    if variavel == 'temp':
        var_col = next((c for c in df.columns if 'temp' in c.lower()), None)
    elif variavel == 'prec':
        var_col = next((c for c in df.columns if 'prec' in c.lower() or 'chirps' in c.lower()), None)
    else:
        var_col = next((c for c in df.columns if 'sis' in c.lower() or 'rad' in c.lower()), None)
    
    # Filtrar por bloco
    mask_bloco = (
        (df[lat_col] >= bloco['bbox_miny']) & 
        (df[lat_col] <= bloco['bbox_maxy']) &
        (df[lon_col] >= bloco['bbox_minx']) & 
        (df[lon_col] <= bloco['bbox_maxx'])
    )
    
    df_filtrado = df[mask_bloco].copy()
    
    if len(df_filtrado) == 0:
        return None
    
    # Converter data
    if not pd.api.types.is_datetime64_any_dtype(df_filtrado[date_col]):
        df_filtrado[date_col] = pd.to_datetime(df_filtrado[date_col])
    
    df_filtrado['ano'] = df_filtrado[date_col].dt.year
    df_filtrado = df_filtrado[(df_filtrado['ano'] >= 1990) & (df_filtrado['ano'] <= 2024)]
    
    # Agregar
    if variavel == 'prec':
        df_anual = df_filtrado.groupby('ano')[var_col].sum().reset_index()
    else:
        df_anual = df_filtrado.groupby('ano')[var_col].mean().reset_index()
    
    df_anual.columns = ['ano', 'valor']
    
    # Converter temperatura
    if variavel == 'temp' and df_anual['valor'].mean() > 100:
        df_anual['valor'] -= 273.15
    
    return df_anual

def plotar_4_blocos(blocos_perda, blocos_ganho, arquivo_parquet, variavel, titulo, ylabel, output_file):
    """Plota 4 linhas - 2 de perda + 2 de ganho"""
    print(f"\n  Extraindo {variavel} para 4 blocos individuais...")
    
    # Extrair séries para cada bloco
    series_blocos = []
    
    # Blocos de perda
    for idx, bloco in blocos_perda.iterrows():
        print(f"    Bloco #{int(bloco['bloco_id'])} (PERDA)...", end=" ")
        df = extrair_serie_temporal_bloco_individual(arquivo_parquet, bloco, variavel)
        if df is not None:
            print(f"{len(df)} anos ✓")
            series_blocos.append({
                'bloco_id': int(bloco['bloco_id']),
                'tipo': 'perda',
                'data': df,
                'pct_perda': bloco['pct_perda']
            })
        else:
            print("Sem dados ✗")
    
    # Blocos de ganho
    for idx, bloco in blocos_ganho.iterrows():
        print(f"    Bloco #{int(bloco['bloco_id'])} (GANHO)...", end=" ")
        df = extrair_serie_temporal_bloco_individual(arquivo_parquet, bloco, variavel)
        if df is not None:
            print(f"{len(df)} anos ✓")
            series_blocos.append({
                'bloco_id': int(bloco['bloco_id']),
                'tipo': 'ganho',
                'data': df,
                'pct_ganho': bloco['pct_ganho']
            })
        else:
            print("Sem dados ✗")
    
    if len(series_blocos) == 0:
        print(f"    ⚠️  Nenhum dado encontrado!")
        return
    
    # Plotar
    import os
    os.makedirs('./resultados', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plotar blocos de perda (vermelhos)
    cores_perda = ['#E74C3C', '#C0392B']  # Vermelho escuro e mais claro
    estilos_perda = ['-', '--']
    
    idx_perda = 0
    for bloco in series_blocos:
        if bloco['tipo'] == 'perda':
            ax.plot(bloco['data']['ano'], bloco['data']['valor'],
                   color=cores_perda[idx_perda % 2],
                   linestyle=estilos_perda[idx_perda % 2],
                   marker='s', linewidth=2.5, markersize=8,
                   label=f'Perda #{idx_perda+1}',
                   zorder=3)
            idx_perda += 1
    
    # Plotar blocos de ganho (verdes)
    cores_ganho = ['#27AE60', '#229954']  # Verde escuro e mais claro
    estilos_ganho = ['-', '--']
    
    idx_ganho = 0
    for bloco in series_blocos:
        if bloco['tipo'] == 'ganho':
            ax.plot(bloco['data']['ano'], bloco['data']['valor'],
                   color=cores_ganho[idx_ganho % 2],
                   linestyle=estilos_ganho[idx_ganho % 2],
                   marker='o', linewidth=2.5, markersize=8,
                   label=f'Ganho #{idx_ganho+1}',
                   zorder=3)
            idx_ganho += 1
    
    # Configurações
    ax.set_title(titulo, fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel('Ano', fontsize=18, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12, loc='best', frameon=True, 
             facecolor='white', edgecolor='gray', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#EBEBEB')
    
    plt.tight_layout()
    plt.savefig(f'./resultados/{output_file}', dpi=300, bbox_inches='tight')
    print(f"  ✓ Salvo: {output_file}")
    plt.close()

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("ANÁLISE DE BLOCOS EXTREMOS - 4 LINHAS POR GRÁFICO")
    print("="*80)
    
    blocos_perda, blocos_ganho = identificar_blocos_extremos(
        tif_1990, tif_2020, CLASSE_SAVANA, TAMANHO_BLOCO
    )
    
    blocos_perda.to_csv('./resultados/blocos_maior_perda_sp.csv', index=False)
    blocos_ganho.to_csv('./resultados/blocos_maior_ganho_sp.csv', index=False)
    
    print(f"\n{'='*80}")
    print("EXTRAINDO SÉRIES TEMPORAIS")
    print('='*80)
    
    # TEMPERATURA
    print("\n[1/3] 🌡️  TEMPERATURA")
    plotar_4_blocos(
        blocos_perda, blocos_ganho,
        arquivo_temp, 'temp',
        'Médias Anuais - Temperatura Média (°C)\nFormação Savânica - RN (1990-2024)',
        'Temperatura Média (°C)',
        'temp_formacao_savânica_rn.png'
    )
    
    # PRECIPITAÇÃO
    print("\n[2/3] 🌧️  PRECIPITAÇÃO")
    plotar_4_blocos(
        blocos_perda, blocos_ganho,
        arquivo_prec, 'prec',
        'Precipitação Total Anual\nFormação Savânica - RN (1990-2024)',
        'Precipitação Total Anual (mm)',
        'prec_formação_savânica_rn.png'
    )
    
    # RADIAÇÃO
    print("\n[3/3] ☀️  RADIAÇÃO")
    plotar_4_blocos(
        blocos_perda, blocos_ganho,
        arquivo_rad, 'rad',
        'Médias Anuais - Radiação Solar Média (W/m²)\nFormação Savânica - RN (1990-2024)',
        'Radiação Solar Média (W/m²)',
        'rad_formação_savânica_rn.png'
    )
    
    print("\n✅ CONCLUÍDO!")