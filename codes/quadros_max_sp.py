import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
# Caminhos dos arquivos de entrada
RASTER_1990 = r"C:\Users\camil\Downloads\mapbiomas\SP_1990_lclu_recortado.tif"
RASTER_2020 = r"C:\Users\camil\Downloads\mapbiomas\SP_2020_lclu_recortado.tif"
SHAPEFILE_GRADE = "grade_analise_SP.shp"

# Pasta de saída
OUTPUT_DIR = r"F:\validacoes"

# Classe de interesse
CLASSE_FLORESTA = 3  # Formação Florestal

# Número de extremos a mostrar (AJUSTÁVEL)
N_EXTREMOS = 2  # 2 maiores perdas + 2 maiores ganhos

# Anos disponíveis (ajuste conforme seus dados)
ANOS_DISPONIVEIS = [1990, 2000, 2010, 2020]

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def calcular_cobertura_por_ano(raster_path, grade, classe_alvo, ano):
    """
    Calcula a cobertura vegetal por célula para um ano específico
    """
    print(f"\n  Processando ano {ano}...")
    
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        
        if grade.crs != raster_crs:
            grade_reprojected = grade.to_crs(raster_crs)
        else:
            grade_reprojected = grade.copy()
    
    with rasterio.open(raster_path) as src:
        resultados = []
        
        for idx, row in grade_reprojected.iterrows():
            cell_id = row['cell_id']
            geom = [row.geometry.__geo_interface__]
            
            try:
                img, _ = mask(src, geom, crop=True, nodata=0)
                img = img[0]
                
                pixels_validos = np.sum(img > 0)
                
                if pixels_validos == 0:
                    continue
                
                pixels_classe = np.sum(img == classe_alvo)
                pct_vegetacao = (pixels_classe / pixels_validos) * 100
                
                resultados.append({
                    'cell_id': cell_id,
                    'ano': ano,
                    'pct_vegetacao': pct_vegetacao
                })
                
                if (idx + 1) % 50 == 0:
                    print(f"    Processadas {idx + 1}/{len(grade_reprojected)} células...")
                
            except Exception as e:
                if "Input shapes do not overlap" not in str(e):
                    print(f"    Erro na célula {cell_id}: {str(e)}")
                continue
        
        return pd.DataFrame(resultados)

def gerar_dados_temporais_completos(grade, rasters_dict, classe_alvo):
    """
    Gera dados de cobertura vegetal para todos os anos disponíveis
    """
    print("\n" + "="*80)
    print("GERANDO DADOS TEMPORAIS DE COBERTURA VEGETAL")
    print("="*80)
    
    dados_completos = []
    
    for ano, raster_path in rasters_dict.items():
        df_ano = calcular_cobertura_por_ano(raster_path, grade, classe_alvo, ano)
        dados_completos.append(df_ano)
    
    df_final = pd.concat(dados_completos, ignore_index=True)
    
    # Calcular mudanças
    df_pivot = df_final.pivot(index='cell_id', columns='ano', values='pct_vegetacao')
    primeiro_ano = min(rasters_dict.keys())
    ultimo_ano = max(rasters_dict.keys())
    
    df_final = df_final.merge(
        df_pivot[[primeiro_ano, ultimo_ano]].reset_index(),
        on='cell_id',
        how='left'
    )
    
    df_final['mudanca_total'] = df_final[ultimo_ano] - df_final[primeiro_ano]
    df_final['vegetacao_inicial'] = df_final[primeiro_ano]
    df_final['vegetacao_final'] = df_final[ultimo_ano]
    
    return df_final

def identificar_blocos_extremos(df_cobertura, n_extremos=2):
    """
    Identifica os n blocos com maior perda e maior ganho de cobertura vegetal
    """
    print(f"\n  Identificando {n_extremos} blocos de cada extremo...")
    
    df_mudanca = df_cobertura.groupby('cell_id').first().reset_index()
    df_mudanca = df_mudanca.sort_values('mudanca_total')
    
    if len(df_mudanca) < n_extremos * 2:
        print(f"    AVISO: Apenas {len(df_mudanca)} células disponíveis!")
        return None
    
    # Pegar os n blocos com MAIOR PERDA (valores mais negativos)
    blocos_maior_perda = df_mudanca.head(n_extremos)
    
    # Pegar os n blocos com MAIOR GANHO (valores mais positivos)
    blocos_maior_ganho = df_mudanca.tail(n_extremos)
    
    print(f"    Total de células: {len(df_mudanca)}")
    print(f"\n    Top {n_extremos} MAIORES PERDAS:")
    for idx, row in blocos_maior_perda.iterrows():
        print(f"      Cell {int(row['cell_id'])}: {row['mudanca_total']:+6.2f}%")
    
    print(f"\n    Top {n_extremos} MAIORES GANHOS:")
    for idx, row in blocos_maior_ganho.iterrows():
        print(f"      Cell {int(row['cell_id'])}: {row['mudanca_total']:+6.2f}%")
    
    return {
        'blocos_perda': blocos_maior_perda,
        'blocos_ganho': blocos_maior_ganho,
        'todos': df_mudanca
    }

def plot_evolucao_cobertura_vegetal(blocos_info, df_cobertura, classe_nome='Formação Florestal'):
    """
    Plota a evolução temporal da cobertura vegetal para múltiplos blocos extremos
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    blocos_perda = blocos_info['blocos_perda']
    blocos_ganho = blocos_info['blocos_ganho']
    
    print(f"\n  Gerando gráfico de evolução temporal...")
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Cores para múltiplas linhas
    cores_perda = ['#8b0000', '#cd5c5c']  # Vermelho escuro, vermelho claro
    cores_ganho = ['#006400', '#228b22']  # Verde escuro, verde claro
    
    # Plotar linhas de PERDA
    for i, (idx, row) in enumerate(blocos_perda.iterrows()):
        cell_id = row['cell_id']
        df_cob = df_cobertura[df_cobertura['cell_id'] == cell_id].sort_values('ano')
        
        ax.plot(df_cob['ano'], df_cob['pct_vegetacao'], 
               marker='s', linewidth=3.5, 
               label=f'Perda #{i+1} (Cell {int(cell_id)})', 
               color=cores_perda[i % len(cores_perda)], 
               alpha=0.85, markersize=11,
               linestyle='--' if i > 0 else '-')
    
    # Plotar linhas de GANHO
    for i, (idx, row) in enumerate(blocos_ganho.iterrows()):
        cell_id = row['cell_id']
        df_cob = df_cobertura[df_cobertura['cell_id'] == cell_id].sort_values('ano')
        
        ax.plot(df_cob['ano'], df_cob['pct_vegetacao'], 
               marker='o', linewidth=3.5, 
               label=f'Ganho #{i+1} (Cell {int(cell_id)})', 
               color=cores_ganho[i % len(cores_ganho)], 
               alpha=0.85, markersize=11,
               linestyle='--' if i > 0 else '-')
    
    # Configurações
    ax.set_xlabel('Ano', fontsize=28, fontweight='bold')
    ax.set_ylabel(f'Cobertura de {classe_nome} (%)', fontsize=28, fontweight='bold')
    ax.set_title(f'Evolução da Cobertura de {classe_nome}\nSP (1990-2020)', 
                fontsize=32, fontweight='bold', pad=25)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(loc='best', fontsize=20, frameon=True, shadow=True, fancybox=True, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'evolucao_cobertura_vegetal_sp.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"    Gráfico salvo: {output_file}")

def estatisticas_descritivas(blocos_info, df_cobertura):
    """
    Exibe estatísticas dos blocos extremos
    """
    print(f"\n{'='*80}")
    print("ESTATÍSTICAS DESCRITIVAS")
    print('='*80)
    
    blocos_perda = blocos_info['blocos_perda']
    blocos_ganho = blocos_info['blocos_ganho']
    
    print("\nMAIORES PERDAS:")
    for i, (idx, row) in enumerate(blocos_perda.iterrows()):
        cell_id = row['cell_id']
        df_cell = df_cobertura[df_cobertura['cell_id'] == cell_id].sort_values('ano')
        
        print(f"\n  Perda #{i+1} (Cell ID: {int(cell_id)}):")
        print(f"    Cobertura inicial: {df_cell.iloc[0]['pct_vegetacao']:.2f}%")
        print(f"    Cobertura final: {df_cell.iloc[-1]['pct_vegetacao']:.2f}%")
        print(f"    Mudança total: {row['mudanca_total']:.2f}%")
        print(f"    Cobertura média: {df_cell['pct_vegetacao'].mean():.2f}%")
    
    print("\n\nMAIORES GANHOS:")
    for i, (idx, row) in enumerate(blocos_ganho.iterrows()):
        cell_id = row['cell_id']
        df_cell = df_cobertura[df_cobertura['cell_id'] == cell_id].sort_values('ano')
        
        print(f"\n  Ganho #{i+1} (Cell ID: {int(cell_id)}):")
        print(f"    Cobertura inicial: {df_cell.iloc[0]['pct_vegetacao']:.2f}%")
        print(f"    Cobertura final: {df_cell.iloc[-1]['pct_vegetacao']:.2f}%")
        print(f"    Mudança total: {row['mudanca_total']:.2f}%")
        print(f"    Cobertura média: {df_cell['pct_vegetacao'].mean():.2f}%")

def plot_mapa_blocos_extremos(blocos_info, df_cobertura, grade, classe_nome='Formação Florestal'):
    """
    Plota mapa mostrando blocos extremos com LEGENDA LATERAL
    """
    from matplotlib.patches import Patch, Rectangle
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    blocos_perda = blocos_info['blocos_perda']
    blocos_ganho = blocos_info['blocos_ganho']
    
    print(f"\n  Gerando mapa dos blocos extremos...")
    
    # Preparar dados
    grade_wgs = grade.to_crs("EPSG:4326")
    df_mudanca = df_cobertura.groupby('cell_id').first().reset_index()
    grade_plot = grade_wgs.merge(df_mudanca[['cell_id', 'mudanca_total']], 
                                  on='cell_id', how='left')
    
    # Criar figura com espaço para legenda lateral
    fig = plt.figure(figsize=(22, 14))
    
    # Grid: mapa ocupa 72% da largura, legenda 28%, espaço mínimo entre eles
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1], wspace=0.05)
    ax_mapa = fig.add_subplot(gs[0])
    ax_legenda = fig.add_subplot(gs[1])
    
    # Plot do mapa base
    grade_plot.plot(column='mudanca_total', cmap='RdYlGn', legend=True, ax=ax_mapa,
                    edgecolor='gray', linewidth=0.5, alpha=0.7,
                    legend_kwds={'label': 'Mudança (%)', 'shrink': 0.7, 'pad': 0.05},
                    missing_kwds={'color': 'lightgray'})
    
    # Cores para bordas
    cores_perda = ['#8b0000', '#cd5c5c']  # Vermelho escuro, vermelho claro
    cores_ganho = ['#006400', '#228b22']  # Verde escuro, verde claro
    
    # Destacar blocos de PERDA (apenas bordas, SEM labels)
    for i, (idx, row) in enumerate(blocos_perda.iterrows()):
        cell_id = row['cell_id']
        grade_wgs[grade_wgs['cell_id'] == cell_id].boundary.plot(
            ax=ax_mapa, edgecolor=cores_perda[i % len(cores_perda)], 
            linewidth=5, linestyle='-' if i == 0 else '--')
    
    # Destacar blocos de GANHO (apenas bordas, SEM labels)
    for i, (idx, row) in enumerate(blocos_ganho.iterrows()):
        cell_id = row['cell_id']
        grade_wgs[grade_wgs['cell_id'] == cell_id].boundary.plot(
            ax=ax_mapa, edgecolor=cores_ganho[i % len(cores_ganho)], 
            linewidth=5, linestyle='-' if i == 0 else '--')
    
    # Configurar mapa
    ax_mapa.set_title(f'{classe_nome} - Mudanças Extremas\nSP (1990-2020)', 
                      fontsize=28, fontweight='bold', pad=20)
    ax_mapa.set_xlabel('Longitude', fontsize=24, fontweight='bold')
    ax_mapa.set_ylabel('Latitude', fontsize=24, fontweight='bold')
    ax_mapa.tick_params(axis='both', which='major', labelsize=20)
    
    # ========================================================================
    # CRIAR LEGENDA LATERAL COM INFORMAÇÕES DOS BLOCOS
    # ========================================================================
    
    ax_legenda.axis('off')
    
    y_pos = 0.95  # Posição vertical inicial (começar mais alto)
    
    # SEÇÃO: MAIORES PERDAS
    ax_legenda.text(0.5, y_pos, 'MAIORES PERDAS', 
                    ha='center', va='top', fontsize=22, fontweight='bold',
                    color='#8b0000', transform=ax_legenda.transAxes)
    y_pos -= 0.06
    
    for i, (idx, row) in enumerate(blocos_perda.iterrows()):
        cell_id = row['cell_id']
        mudanca = row['mudanca_total']
        df_cell = df_cobertura[df_cobertura['cell_id'] == cell_id].sort_values('ano')
        cob_inicial = df_cell.iloc[0]['pct_vegetacao']
        cob_final = df_cell.iloc[-1]['pct_vegetacao']
        
        cor = cores_perda[i % len(cores_perda)]
        linestyle = '-' if i == 0 else '--'
        
        # Retângulo colorido (mais à esquerda e maior)
        rect = Rectangle((0.02, y_pos - 0.035), 0.10, 0.028, 
                         facecolor=cor, edgecolor='black', linewidth=2.5,
                         linestyle=linestyle, transform=ax_legenda.transAxes)
        ax_legenda.add_patch(rect)
        
        # Texto com informações (mais próximo do retângulo)
        texto = f"Perda #{i+1}\n"
        texto += f"Cell ID: {int(cell_id)}\n"
        texto += f"Mudança: {mudanca:+.1f}%\n"
        texto += f"1990: {cob_inicial:.1f}%\n"
        texto += f"2020: {cob_final:.1f}%"
        
        ax_legenda.text(0.14, y_pos, texto,
                       ha='left', va='top', fontsize=18,
                       transform=ax_legenda.transAxes,
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='white', 
                                edgecolor=cor,
                                linewidth=2.5,
                                alpha=0.95))
        
        y_pos -= 0.18  # Espaço para próximo bloco
    
    y_pos -= 0.04  # Espaço extra entre seções
    
    # SEÇÃO: MAIORES GANHOS
    ax_legenda.text(0.5, y_pos, 'MAIORES GANHOS', 
                    ha='center', va='top', fontsize=22, fontweight='bold',
                    color='#006400', transform=ax_legenda.transAxes)
    y_pos -= 0.06
    
    for i, (idx, row) in enumerate(blocos_ganho.iterrows()):
        cell_id = row['cell_id']
        mudanca = row['mudanca_total']
        df_cell = df_cobertura[df_cobertura['cell_id'] == cell_id].sort_values('ano')
        cob_inicial = df_cell.iloc[0]['pct_vegetacao']
        cob_final = df_cell.iloc[-1]['pct_vegetacao']
        
        cor = cores_ganho[i % len(cores_ganho)]
        linestyle = '-' if i == 0 else '--'
        
        # Retângulo colorido (mais à esquerda e maior)
        rect = Rectangle((0.02, y_pos - 0.035), 0.10, 0.028, 
                         facecolor=cor, edgecolor='black', linewidth=2.5,
                         linestyle=linestyle, transform=ax_legenda.transAxes)
        ax_legenda.add_patch(rect)
        
        # Texto com informações (mais próximo do retângulo)
        texto = f"Ganho #{i+1}\n"
        texto += f"Cell ID: {int(cell_id)}\n"
        texto += f"Mudança: {mudanca:+.1f}%\n"
        texto += f"1990: {cob_inicial:.1f}%\n"
        texto += f"2020: {cob_final:.1f}%"
        
        ax_legenda.text(0.14, y_pos, texto,
                       ha='left', va='top', fontsize=18,
                       transform=ax_legenda.transAxes,
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='white', 
                                edgecolor=cor,
                                linewidth=2.5,
                                alpha=0.95))
        
        y_pos -= 0.18  # Espaço para próximo bloco
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'mapa_blocos_extremos_sp.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"    Mapa salvo: {output_file}")

# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("ANÁLISE DE EVOLUÇÃO DA COBERTURA VEGETAL - SP")
    print(f"Mostrando {N_EXTREMOS} blocos extremos de cada lado")
    print("="*80)
    
    # [1] Carregar grade
    print("\n[1/4] Carregando grade...")
    grade = gpd.read_file(SHAPEFILE_GRADE)
    print(f"  Grade carregada: {len(grade)} células")
    
    # [2] Gerar dados temporais
    rasters_disponiveis = {
        1990: RASTER_1990,
        2020: RASTER_2020,
        2000: r"C:\Users\camil\Downloads\mapbiomas\SP_2000_lclu_recortado.tif",
        2010: r"C:\Users\camil\Downloads\mapbiomas\SP_2010_lclu_recortado.tif",
    }
    
    print("\n[2/4] Gerando dados temporais de cobertura...")
    df_cobertura = gerar_dados_temporais_completos(grade, rasters_disponiveis, CLASSE_FLORESTA)
    
    # Salvar dados
    output_csv = os.path.join(OUTPUT_DIR, "dados_cobertura_vegetal_sp.csv")
    df_cobertura.to_csv(output_csv, index=False)
    print(f"\n  Dados salvos: {output_csv}")
    print(f"  Total de registros: {len(df_cobertura)}")
    
    # [3] Identificar blocos extremos
    print("\n[3/4] Identificando blocos extremos...")
    blocos_info = identificar_blocos_extremos(df_cobertura, n_extremos=N_EXTREMOS)
    
    if blocos_info is None:
        print("\nERRO: Dados insuficientes!")
        import sys
        sys.exit(1)
    
    # [4] Gerar gráficos
    print("\n[4/4] Gerando visualizações...")
    estatisticas_descritivas(blocos_info, df_cobertura)
    plot_evolucao_cobertura_vegetal(blocos_info, df_cobertura, classe_nome='Formação Florestal')
    plot_mapa_blocos_extremos(blocos_info, df_cobertura, grade, classe_nome='Formação Florestal')
    
    print("\n" + "="*80)
    print("ANÁLISE CONCLUÍDA!")
    print("="*80)
    print(f"\nArquivos gerados em: {OUTPUT_DIR}")
    print(f"  - dados_cobertura_vegetal_sp.csv")
    print(f"  - evolucao_cobertura_vegetal_sp.png")
    print(f"  - mapa_blocos_extremos_sp.png")
    print(f"\nBlocos analisados ({N_EXTREMOS} de cada extremo):")
    
    print(f"\n  MAIORES PERDAS:")
    for i, (idx, row) in enumerate(blocos_info['blocos_perda'].iterrows()):
        print(f"    #{i+1}: Cell ID {int(row['cell_id'])} ({row['mudanca_total']:+.2f}%)")
    
    print(f"\n  MAIORES GANHOS:")
    for i, (idx, row) in enumerate(blocos_info['blocos_ganho'].iterrows()):
        print(f"    #{i+1}: Cell ID {int(row['cell_id'])} ({row['mudanca_total']:+.2f}%)")