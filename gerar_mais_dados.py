#!/usr/bin/env python3
"""
Gerador de Mais Dados para Novas Classes
Gera dados adicionais para as classes que têm poucas amostras
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

class GeradorMaisDados:
    def __init__(self):
        # Letras que precisam de mais dados (as novas)
        self.letras_novas = [
            'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'M', 'N', 
            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Z'
        ]
        
        # Padrões de gestos baseados em Libras reais
        self.padroes_gestos = {
            'D': {'dedos': [1, 1, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'E': {'dedos': [0, 0, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'F': {'dedos': [1, 1, 0, 0, 1], 'angulo': 0, 'posicao': 'centro'},
            'G': {'dedos': [1, 1, 0, 0, 0], 'angulo': 15, 'posicao': 'lateral'},
            'H': {'dedos': [1, 1, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'I': {'dedos': [0, 0, 0, 0, 1], 'angulo': 0, 'posicao': 'centro'},
            'J': {'dedos': [0, 0, 0, 0, 1], 'angulo': 0, 'posicao': 'centro'},
            'K': {'dedos': [1, 1, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'M': {'dedos': [0, 0, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'N': {'dedos': [0, 0, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'O': {'dedos': [0, 0, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'P': {'dedos': [1, 1, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'Q': {'dedos': [1, 1, 0, 0, 0], 'angulo': 15, 'posicao': 'lateral'},
            'R': {'dedos': [1, 1, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'S': {'dedos': [0, 0, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'T': {'dedos': [0, 0, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'U': {'dedos': [1, 1, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'V': {'dedos': [1, 1, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'W': {'dedos': [1, 1, 1, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'X': {'dedos': [1, 0, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'Z': {'dedos': [1, 1, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'}
        }
        
        self.dados_adicionais = []
    
    def gerar_landmarks_sinteticos(self, letra, variacao=0):
        """Gera landmarks sintéticos baseados no padrão da letra"""
        padrao = self.padroes_gestos[letra]
        
        # Landmarks base (21 pontos da mão)
        landmarks_base = np.zeros((21, 3))
        
        # Pulso (ponto 0) - sempre na origem
        landmarks_base[0] = [0, 0, 0]
        
        # Base dos dedos (pontos 1, 5, 9, 13, 17)
        landmarks_base[1] = [-0.05, -0.02, 0]  # Polegar
        landmarks_base[5] = [-0.03, -0.05, 0]   # Indicador
        landmarks_base[9] = [0, -0.05, 0]        # Médio
        landmarks_base[13] = [0.03, -0.05, 0]    # Anelar
        landmarks_base[17] = [0.05, -0.02, 0]   # Mindinho
        
        # Pontas dos dedos baseadas no padrão
        dedos_config = padrao['dedos']
        
        # Polegar (pontos 2, 3, 4)
        if dedos_config[0]:  # Polegar levantado
            landmarks_base[2] = [-0.08, -0.05, 0]
            landmarks_base[3] = [-0.12, -0.08, 0]
            landmarks_base[4] = [-0.15, -0.1, 0]
        else:  # Polegar abaixado
            landmarks_base[2] = [-0.06, -0.01, 0]
            landmarks_base[3] = [-0.07, 0.01, 0]
            landmarks_base[4] = [-0.08, 0.03, 0]
        
        # Indicador (pontos 6, 7, 8)
        if dedos_config[1]:  # Indicador levantado
            landmarks_base[6] = [-0.05, -0.1, 0]
            landmarks_base[7] = [-0.05, -0.15, 0]
            landmarks_base[8] = [-0.05, -0.2, 0]
        else:  # Indicador abaixado
            landmarks_base[6] = [-0.03, -0.06, 0]
            landmarks_base[7] = [-0.03, -0.07, 0]
            landmarks_base[8] = [-0.03, -0.08, 0]
        
        # Médio (pontos 10, 11, 12)
        if dedos_config[2]:  # Médio levantado
            landmarks_base[10] = [0, -0.1, 0]
            landmarks_base[11] = [0, -0.15, 0]
            landmarks_base[12] = [0, -0.2, 0]
        else:  # Médio abaixado
            landmarks_base[10] = [0, -0.06, 0]
            landmarks_base[11] = [0, -0.07, 0]
            landmarks_base[12] = [0, -0.08, 0]
        
        # Anelar (pontos 14, 15, 16)
        if dedos_config[3]:  # Anelar levantado
            landmarks_base[14] = [0.03, -0.1, 0]
            landmarks_base[15] = [0.03, -0.15, 0]
            landmarks_base[16] = [0.03, -0.2, 0]
        else:  # Anelar abaixado
            landmarks_base[14] = [0.03, -0.06, 0]
            landmarks_base[15] = [0.03, -0.07, 0]
            landmarks_base[16] = [0.03, -0.08, 0]
        
        # Mindinho (pontos 18, 19, 20)
        if dedos_config[4]:  # Mindinho levantado
            landmarks_base[18] = [0.05, -0.1, 0]
            landmarks_base[19] = [0.05, -0.15, 0]
            landmarks_base[20] = [0.05, -0.2, 0]
        else:  # Mindinho abaixado
            landmarks_base[18] = [0.05, -0.06, 0]
            landmarks_base[19] = [0.05, -0.07, 0]
            landmarks_base[20] = [0.05, -0.08, 0]
        
        # Aplicar rotação baseada no ângulo
        angulo_rad = np.radians(padrao['angulo'])
        cos_a, sin_a = np.cos(angulo_rad), np.sin(angulo_rad)
        
        # Matriz de rotação 2D
        for i in range(len(landmarks_base)):
            x, y, z = landmarks_base[i]
            landmarks_base[i] = [
                x * cos_a - y * sin_a,
                x * sin_a + y * cos_a,
                z
            ]
        
        # Adicionar variações aleatórias mais realistas
        if variacao > 0:
            # Variação diferente para cada tipo de ponto
            for i in range(len(landmarks_base)):
                if i == 0:  # Pulso - menos variação
                    ruido = np.random.normal(0, variacao * 0.5, 3)
                elif i in [1, 5, 9, 13, 17]:  # Base dos dedos - variação média
                    ruido = np.random.normal(0, variacao, 3)
                else:  # Pontas dos dedos - mais variação
                    ruido = np.random.normal(0, variacao * 1.5, 3)
                
                landmarks_base[i] += ruido
        
        # Normalizar em relação ao pulso
        pulso = landmarks_base[0].copy()
        for i in range(len(landmarks_base)):
            landmarks_base[i] -= pulso
        
        # Converter para formato de lista (63 valores)
        pontos = []
        for landmark in landmarks_base:
            pontos.extend([landmark[0], landmark[1], landmark[2]])
        
        return pontos
    
    def gerar_dados_letra(self, letra, num_amostras=200):
        """Gera dados sintéticos para uma letra específica"""
        print(f"🎯 Gerando {num_amostras} amostras adicionais para letra {letra}...")
        
        dados_letra = []
        
        for i in range(num_amostras):
            # Gerar landmarks com variação mais realista
            variacao = np.random.uniform(0.02, 0.05)  # Variação maior para mais diversidade
            landmarks = self.gerar_landmarks_sinteticos(letra, variacao)
            
            if len(landmarks) == 63:
                dados_letra.append({
                    'label': letra,
                    **{f'point_{j}': landmarks[j] for j in range(63)}
                })
        
        print(f"✅ Gerados {len(dados_letra)} dados adicionais para {letra}")
        return dados_letra
    
    def gerar_todos_dados_adicionais(self):
        """Gera dados adicionais para todas as letras novas"""
        print("🚀 Iniciando geração de dados adicionais...")
        
        todos_dados = []
        
        for letra in self.letras_novas:
            dados_letra = self.gerar_dados_letra(letra, 200)  # 200 amostras por letra
            todos_dados.extend(dados_letra)
        
        self.dados_adicionais = todos_dados
        print(f"\n✅ Geração concluída! Total: {len(todos_dados)} dados adicionais")
        
        return todos_dados
    
    def salvar_dados_expandidos(self, nome_arquivo='gestos_libras_final.csv'):
        """Salva dados expandidos"""
        if not self.dados_adicionais:
            print("❌ Nenhum dado adicional para salvar")
            return
        
        try:
            # Carregar dados existentes
            dados_existentes = []
            if os.path.exists('gestos_libras_expandido.csv'):
                df_existente = pd.read_csv('gestos_libras_expandido.csv')
                dados_existentes = df_existente.to_dict('records')
                print(f"📊 Carregados {len(dados_existentes)} dados existentes")
            
            # Combinar dados
            todos_dados = dados_existentes + self.dados_adicionais
            
            # Criar DataFrame
            df = pd.DataFrame(todos_dados)
            
            # Salvar
            df.to_csv(nome_arquivo, index=False)
            
            print(f"✅ Dados salvos em {nome_arquivo}")
            print(f"📊 Total de amostras: {len(todos_dados)}")
            print(f"📈 Novas amostras: {len(self.dados_adicionais)}")
            
            # Mostrar distribuição
            print("\n📊 Distribuição por classe:")
            distribuicao = df['label'].value_counts().sort_index()
            print(distribuicao)
            
            # Mostrar estatísticas
            print(f"\n📈 Estatísticas:")
            print(f"   - Classes únicas: {len(df['label'].unique())}")
            print(f"   - Amostras por classe (média): {len(df) / len(df['label'].unique()):.1f}")
            print(f"   - Amostras por classe (min): {distribuicao.min()}")
            print(f"   - Amostras por classe (max): {distribuicao.max()}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar dados: {e}")
    
    def treinar_modelo_final(self):
        """Treina modelo final com todos os dados"""
        try:
            print("\n🧠 Treinando modelo final com dados expandidos...")
            
            # Carregar dados
            df = pd.read_csv('gestos_libras_final.csv')
            feature_columns = [col for col in df.columns if col != 'label']
            X = df[feature_columns].values
            y = df['label'].values
            
            print(f"📊 Dados: {len(df)} amostras, {len(feature_columns)} features")
            print(f"🏷️ Classes: {sorted(df['label'].unique())}")
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Treinar modelo com parâmetros otimizados
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Avaliar
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            
            print(f"📈 Acurácia treino: {train_acc:.2%}")
            print(f"📈 Acurácia teste: {test_acc:.2%}")
            
            # Salvar modelo
            os.makedirs('modelos', exist_ok=True)
            with open('modelos/modelo_libras_final.pkl', 'wb') as f:
                pickle.dump(model, f)
            
            # Salvar informações
            model_info = {
                'classes': model.classes_.tolist(),
                'n_features': len(feature_columns),
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'n_samples': len(df),
                'vocabulary_type': 'final_expanded',
                'created_at': datetime.now().isoformat(),
                'distribution': df['label'].value_counts().to_dict()
            }
            
            with open('modelos/modelo_info_final.pkl', 'wb') as f:
                pickle.dump(model_info, f)
            
            print("✅ Modelo final salvo!")
            return True
            
        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
            return False

def main():
    print("🚀 TraduLibras - Gerador de Mais Dados v1.0")
    print("📚 Gerando dados adicionais para as novas classes...")
    
    gerador = GeradorMaisDados()
    
    # Gerar dados adicionais
    dados = gerador.gerar_todos_dados_adicionais()
    
    if dados:
        # Salvar dados expandidos
        gerador.salvar_dados_expandidos()
        
        # Treinar modelo final
        gerador.treinar_modelo_final()
        
        print("\n🎉 Processo concluído!")
        print("📁 Arquivos criados:")
        print("   - gestos_libras_final.csv")
        print("   - modelos/modelo_libras_final.pkl")
        print("   - modelos/modelo_info_final.pkl")
        
        print("\n📊 Resumo Final:")
        print(f"   - Total de classes: 26")
        print(f"   - Amostras por classe: ~250-650")
        print(f"   - Total de amostras: ~{len(dados) + 3783}")
    else:
        print("❌ Nenhum dado foi gerado")

if __name__ == "__main__":
    main()
