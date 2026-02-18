import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import warnings
import gc
import pyarrow.parquet as pq
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
tif_1990 = r"C:\Users\camil\Downloads\mapbiomas\SP_1990_lclu_recortado.tif"
tif_2020 = r"C:\Users\camil\Downloads\mapbiomas\SP_2020_lclu_recortado.tif"

arquivo_temp = r"F:\sp\ERA5_TemperaturaMedia_SP_1990_2024.parquet"
arquivo_prec = r"F:\sp\CHIRPS_SP_diaria_por_pixel_1990_2024.parquet"
arquivo_rad = r"F:\sp\SIS_SP.parquet"

CLASSE_FLORESTA = 3
TAMANHO_BLOCO = 200
CHUNK_SIZE = 1000000  # 1M linhas por chunk

def identificar_blocos_extremos_OTIMIZADO(tif_1990, tif_2020, classe_alvo, tamanho_bloco=200):
    """Versão SUPER OTIMIZADA"""
    print(f"\n{'='*80}")
    print(f"Identificando blocos extremos - Classe {classe_alvo}")
    print('='*80)
    
    with rasterio.open(tif_1990) as src:
        shape = src.shape
        transform = src.transform
        
        print(f"  Shape: {shape}")
        
        n_blocos_y = shape[0] // tamanho_bloco
        n_blocos_x = shape[1] // tamanho_bloco
        total_blocos = n_blocos_y * n_blocos_x
        
        print(f"  Grade: {n_blocos_y} x {n_blocos_x} = {total_blocos} blocos")
        
        blocos_info = []
        contador = 0
        
        for i in range(n_blocos_y):
            y_start_chunk = i * tamanho_bloco
            y_end_chunk = (i + 1) * tamanho_bloco
            
            img_1990_chunk = src.read(1, window=((y_start_chunk, y_end_chunk), (0, shape[1])))
            
            with rasterio.open(tif_2020) as src2020:
                img_2020_chunk = src2020.read(1, window=((y_start_chunk, y_end_chunk), (0, shape[1])))
            
            mask_1990 = (img_1990_chunk == classe_alvo).astype(np.int8)
            mask_2020 = (img_2020_chunk == classe_alvo).astype(np.int8)
            
            del img_1990_chunk, img_2020_chunk
            
            mudanca_chunk = mask_2020 - mask_1990
            del mask_1990, mask_2020
            
            for j in range(n_blocos_x):
                contador += 1
                if contador % 100 == 0:
                    print(f"  Bloco {contador}/{total_blocos} ({(contador/total_blocos*100):.1f}%)...", end='\r')
                
                x_start = j * tamanho_bloco
                x_end = (j + 1) * tamanho_bloco
                
                bloco_mudanca = mudanca_chunk[:, x_start:x_end]
                
                total_pixels = bloco_mudanca.size
                pixels_perda = np.sum(bloco_mudanca == -1)
                pixels_ganho = np.sum(bloco_mudanca == 1)
                
                pct_perda = (pixels_perda / total_pixels) * 100
                pct_ganho = (pixels_ganho / total_pixels) * 100
                mudanca_liquida = pct_ganho - pct_perda
                
                y_start_global = y_start_chunk
                y_end_global = y_end_chunk
                
                coords_cantos = []
                for y_idx in [y_start_global, y_end_global]:
                    for x_idx in [x_start, x_end]:
                        lon, lat = rasterio.transform.xy(transform, y_idx, x_idx)
                        coords_cantos.append((lon, lat))
                
                lons = [c[0] for c in coords_cantos]
                lats = [c[1] for c in coords_cantos]
                
                buffer = 0.15
                
                blocos_info.append({
                    'bloco_id': int(i * n_blocos_x + j),
                    'bbox_minx': min(lons) - buffer,
                    'bbox_maxx': max(lons) + buffer,
                    'bbox_miny': min(lats) - buffer,
                    'bbox_maxy': max(lats) + buffer,
                    'pct_perda': pct_perda,
                    'pct_ganho': pct_ganho,
                    'mudanca_liquida': mudanca_liquida
                })
            
            del mudanca_chunk
            gc.collect()
        
        print(f"\n  {total_blocos} blocos processados")
    
    df_blocos = pd.DataFrame(blocos_info)
    df_blocos_sorted = df_blocos.sort_values('mudanca_liquida')
    
    blocos_maior_perda = df_blocos_sorted.head(2).copy()
    blocos_maior_ganho = df_blocos_sorted.tail(2).copy()
    
    print(f"\n  BLOCOS COM MAIOR PERDA:")
    for idx, row in blocos_maior_perda.iterrows():
        print(f"    Bloco #{int(row['bloco_id']):4d} | Perda: {row['pct_perda']:5.1f}%")
    
    print(f"\n  BLOCOS COM MAIOR GANHO:")
    for idx, row in blocos_maior_ganho.iterrows():
        print(f"    Bloco #{int(row['bloco_id']):4d} | Ganho: {row['pct_ganho']:5.1f}%")
    
    return blocos_maior_perda, blocos_maior_ganho

def get_parquet_columns(arquivo_parquet):
    """
    Lê apenas os METADADOS do Parquet (sem carregar dados)
    """
    parquet_file = pq.ParquetFile(arquivo_parquet)
    schema = parquet_file.schema_arrow
    colunas = schema.names
    return colunas

def extrair_serie_temporal_CHUNKED(arquivo_parquet, bloco, variavel='temp'):
    """
    Extrai série temporal processando Parquet em CHUNKS
    SEM carregar o arquivo inteiro na memória
    """
    
    print(f"      Lendo metadados do arquivo...")
    
    # LER APENAS METADADOS (não carrega dados!)
    colunas_disponiveis = get_parquet_columns(arquivo_parquet)
    
    # Identificar colunas necessárias
    lat_col = next((c for c in colunas_disponiveis if 'lat' in c.lower()), None)
    lon_col = next((c for c in colunas_disponiveis if 'lon' in c.lower()), None)
    date_col = next((c for c in colunas_disponiveis if any(x in c.lower() for x in ['date', 'data', 'time'])), None)
    
    if variavel == 'temp':
        var_col = next((c for c in colunas_disponiveis if 'temp' in c.lower()), None)
    elif variavel == 'prec':
        var_col = next((c for c in colunas_disponiveis if 'prec' in c.lower() or 'chirps' in c.lower()), None)
    else:
        var_col = next((c for c in colunas_disponiveis if 'sis' in c.lower() or 'rad' in c.lower()), None)
    
    colunas_carregar = [lat_col, lon_col, date_col, var_col]
    
    print(f"      Colunas: {', '.join(colunas_carregar)}")
    print(f"      Processando em chunks...")
    
    # Usar ParquetFile para iterar por batches
    parquet_file = pq.ParquetFile(arquivo_parquet)
    
    chunks_filtrados = []
    chunk_num = 0
    total_registros = 0
    registros_filtrados = 0
    
    # Iterar pelos batches do Parquet
    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE, columns=colunas_carregar):
        chunk_num += 1
        
        # Converter batch para DataFrame
        chunk = batch.to_pandas()
        total_registros += len(chunk)
        
        # Filtrar geograficamente
        mask_geo = (
            (chunk[lat_col] >= bloco['bbox_miny']) & 
            (chunk[lat_col] <= bloco['bbox_maxy']) &
            (chunk[lon_col] >= bloco['bbox_minx']) & 
            (chunk[lon_col] <= bloco['bbox_maxx'])
        )
        
        chunk_filtrado = chunk[mask_geo].copy()
        
        if len(chunk_filtrado) > 0:
            chunks_filtrados.append(chunk_filtrado)
            registros_filtrados += len(chunk_filtrado)
        
        del chunk, chunk_filtrado, batch
        
        if chunk_num % 5 == 0:
            print(f"      Chunk {chunk_num} | Total: {total_registros:,} | Filtrados: {registros_filtrados:,}", end='\r')
        
        gc.collect()
    
    print(f"\nProcessados: {total_registros:,} | Filtrados: {registros_filtrados:,}")
    
    if len(chunks_filtrados) == 0:
        return None
    
    # Concatenar chunks
    print(f"      Concatenando {len(chunks_filtrados)} chunks...")
    df_filtrado = pd.concat(chunks_filtrados, ignore_index=True)
    
    del chunks_filtrados
    gc.collect()
    
    # Processar data
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
    
    if variavel == 'temp' and df_anual['valor'].mean() > 100:
        df_anual['valor'] -= 273.15
    
    del df_filtrado
    gc.collect()
    
    return df_anual

def plotar_4_blocos(blocos_perda, blocos_ganho, arquivo_parquet, variavel, titulo, ylabel, output_file):
    """Plota 4 linhas"""
    print(f"\n  Extraindo {variavel}...")
    
    series_blocos = []
    
    # Blocos de perda
    for idx, bloco in blocos_perda.iterrows():
        print(f"\n    Bloco #{int(bloco['bloco_id'])} (PERDA):")
        df = extrair_serie_temporal_CHUNKED(arquivo_parquet, bloco, variavel)
        if df is not None:
            print(f"{len(df)} anos | Média: {df['valor'].mean():.2f}")
            series_blocos.append({
                'bloco_id': int(bloco['bloco_id']),
                'tipo': 'perda',
                'data': df,
                'pct_perda': bloco['pct_perda']
            })
        else:
            print(f"      ✗ Sem dados")
    
    # Blocos de ganho
    for idx, bloco in blocos_ganho.iterrows():
        print(f"\n    Bloco #{int(bloco['bloco_id'])} (GANHO):")
        df = extrair_serie_temporal_CHUNKED(arquivo_parquet, bloco, variavel)
        if df is not None:
            print(f"{len(df)} anos | Média: {df['valor'].mean():.2f}")
            series_blocos.append({
                'bloco_id': int(bloco['bloco_id']),
                'tipo': 'ganho',
                'data': df,
                'pct_ganho': bloco['pct_ganho']
            })
        else:
            print(f"      ✗ Sem dados")
    
    if len(series_blocos) == 0:
        print(f"Nenhum dado encontrado!")
        return
    
    # Plotar
    import os
    os.makedirs('./resultados', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    cores_perda = ['#E74C3C', '#C0392B']
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
    
    cores_ganho = ['#27AE60', '#229954']
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
    
    ax.set_title(titulo, fontsize=32, fontweight='bold', pad=20)
    ax.set_xlabel('Ano', fontsize=26, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=26, fontweight='bold')
    ax.tick_params(axis='both', labelsize=22)
    ax.legend(fontsize=20, loc='best', frameon=True, 
             facecolor='white', edgecolor='gray', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#EBEBEB')
    
    plt.tight_layout()
    plt.savefig(f'./resultados/{output_file}', dpi=300, bbox_inches='tight')
    print(f"\n  Salvo: {output_file}")
    plt.close()

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("ANÁLISE SP - FORMAÇÃO FLORESTAL")
    print("="*80)
    
    blocos_perda, blocos_ganho = identificar_blocos_extremos_OTIMIZADO(
        tif_1990, tif_2020, CLASSE_FLORESTA, TAMANHO_BLOCO
    )
    
    blocos_perda.to_csv('./resultados/blocos_maior_perda_sp.csv', index=False)
    blocos_ganho.to_csv('./resultados/blocos_maior_ganho_sp.csv', index=False)
    
    print(f"\n{'='*80}")
    print("EXTRAINDO SÉRIES TEMPORAIS")
    print('='*80)
    
    # TEMPERATURA
    print("\n[1/3] TEMPERATURA")
    plotar_4_blocos(
        blocos_perda, blocos_ganho,
        arquivo_temp, 'temp',
        'Médias Anuais - Temperatura Média (°C)\nFormação Florestal - SP (1990-2024)',
        'Temperatura Média (°C)',
        'temp_formacao_florestal_sp.png'
    )
    
    # PRECIPITAÇÃO
    print("\n[2/3] PRECIPITAÇÃO")
    plotar_4_blocos(
        blocos_perda, blocos_ganho,
        arquivo_prec, 'prec',
        'Precipitação Total Anual\nFormação Florestal - SP (1990-2024)',
        'Precipitação Total Anual (mm)',
        'prec_formacao_florestal_sp.png'
    )
    
    # RADIAÇÃO
    print("\n[3/3] RADIAÇÃO")
    plotar_4_blocos(
        blocos_perda, blocos_ganho,
        arquivo_rad, 'rad',
        'Médias Anuais - Radiação Solar Média (W/m²)\nFormação Florestal - SP (1990-2024)',
        'Radiação Solar Média (W/m²)',
        'rad_formacao_florestal_sp.png'
    )
    
    print("\n CONCLUÍDO!")
