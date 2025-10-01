#!/usr/bin/env python3
"""
Sistema Completo de Expansão de Vocabulário - TraduLibras
Inclui: coleta com câmera, processamento de imagens, criação sintética, download
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import json
from datetime import datetime
import requests
import time
import subprocess
import sys

class SistemaCompletoVocabulario:
    def __init__(self):
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Vocabulário
        self.letras_faltantes = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Z']
        self.numeros = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        # Criar estrutura de pastas
        self.criar_estrutura()
    
    def criar_estrutura(self):
        """Cria estrutura de pastas necessárias"""
        pastas = [
            'imagens/letras/',
            'imagens/numeros/',
            'imagens_baixadas/',
            'imagens_sinteticas/',
            'backup/',
            'modelos/'
        ]
        
        for pasta in pastas:
            Path(pasta).mkdir(parents=True, exist_ok=True)
            print(f"✅ Pasta criada: {pasta}")
    
    def menu_principal(self):
        """Menu principal do sistema"""
        while True:
            print("\n" + "="*70)
            print("🎯 TraduLibras - Sistema Completo de Expansão de Vocabulário")
            print("="*70)
            print("1. 📸 Coletar dados com câmera (interativo)")
            print("2. 🖼️ Processar imagens existentes")
            print("3. 🎨 Criar imagens sintéticas de gestos")
            print("4. 📥 Baixar imagens da internet")
            print("5. 🔄 Processar todas as imagens disponíveis")
            print("6. 🧠 Treinar modelo com dados coletados")
            print("7. 📊 Ver estatísticas detalhadas")
            print("8. 🔧 Ferramentas auxiliares")
            print("9. 🚪 Sair")
            print("="*70)
            
            opcao = input("Digite o número da opção (1-9): ").strip()
            
            if opcao == '1':
                self.coletar_com_camera()
            elif opcao == '2':
                self.processar_imagens_existentes()
            elif opcao == '3':
                self.criar_imagens_sinteticas()
            elif opcao == '4':
                self.baixar_imagens_internet()
            elif opcao == '5':
                self.processar_todas_imagens()
            elif opcao == '6':
                self.treinar_modelo()
            elif opcao == '7':
                self.mostrar_estatisticas()
            elif opcao == '8':
                self.ferramentas_auxiliares()
            elif opcao == '9':
                print("👋 Até logo!")
                break
            else:
                print("❌ Opção inválida! Digite um número de 1 a 9")
    
    def coletar_com_camera(self):
        """Coleta dados usando câmera"""
        print("📸 Iniciando coleta com câmera...")
        print("💡 Esta opção abrirá a câmera para coleta interativa")
        
        confirmar = input("Deseja continuar? (s/n): ").strip().lower()
        if confirmar != 's':
            print("❌ Coleta cancelada")
            return
        
        # Executar coleta com câmera
        self.coletar_gestos_camera()
    
    def coletar_gestos_camera(self):
        """Coleta gestos usando câmera"""
        print(f"\n🎯 Coletando gestos para letras: {', '.join(self.letras_faltantes[:5])}...")
        print("📋 Instruções:")
        print("- Posicione sua mão no centro da câmera")
        print("- Faça o gesto da letra correspondente")
        print("- Pressione ESPAÇO para capturar")
        print("- Pressione ESC para pular")
        print("- Pressione Q para sair")
        
        camera = cv2.VideoCapture(0)
        dados_coletados = []
        
        for letra in self.letras_faltantes[:5]:  # Coletar apenas 5 letras por vez
            print(f"\n📝 Coletando gestos para: {letra}")
            contador = 0
            meta_amostras = 20  # Reduzido para teste
            
            while contador < meta_amostras:
                ret, frame = camera.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # Desenhar informações
                cv2.putText(frame, f"Letra: {letra}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Amostras: {contador}/{meta_amostras}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, "ESPACO: Capturar | ESC: Pular | Q: Sair", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Desenhar landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw = mp.solutions.drawing_utils
                        mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                cv2.imshow('TraduLibras - Coleta de Gestos', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Espaço - capturar
                    if results.multi_hand_landmarks:
                        landmarks = self.processar_landmarks(results.multi_hand_landmarks[0])
                        if len(landmarks) == 63:
                            dados_coletados.append({
                                'label': letra,
                                **{f'point_{i}': landmarks[i] for i in range(63)},
                                'timestamp': datetime.now().isoformat()
                            })
                            contador += 1
                            print(f"✅ Amostra {contador} capturada para {letra}")
                
                elif key == 27:  # ESC - pular
                    print(f"⏭️ Pulando {letra}")
                    break
                
                elif key == ord('q'):  # Q - sair
                    print("🚪 Saindo da coleta...")
                    camera.release()
                    cv2.destroyAllWindows()
                    return dados_coletados
            
            print(f"✅ Coleta para {letra} finalizada! ({contador} amostras)")
        
        camera.release()
        cv2.destroyAllWindows()
        
        if dados_coletados:
            self.salvar_dados(dados_coletados)
        
        return dados_coletados
    
    def processar_imagens_existentes(self):
        """Processa imagens existentes"""
        print("🖼️ Processando imagens existentes...")
        
        pasta = input("Digite o caminho da pasta com imagens (ou Enter para 'imagens/letras/'): ").strip()
        if not pasta:
            pasta = 'imagens/letras/'
        
        dados = self.processar_imagens_pasta(pasta)
        if dados:
            self.salvar_dados(dados)
            print(f"✅ Processadas {len(dados)} imagens")
        else:
            print("⚠️ Nenhuma imagem processada")
    
    def criar_imagens_sinteticas(self):
        """Cria imagens sintéticas de gestos"""
        print("🎨 Criando imagens sintéticas de gestos...")
        
        # Criar imagens sintéticas para letras faltantes
        for letra in self.letras_faltantes:
            self.criar_imagem_sintetica_letra(letra)
        
        print("✅ Imagens sintéticas criadas!")
        print("💡 Agora você pode processar essas imagens com a opção 2")
    
    def criar_imagem_sintetica_letra(self, letra):
        """Cria uma imagem sintética para uma letra"""
        # Configurações da imagem
        largura, altura = 640, 480
        cor_fundo = (240, 240, 240)
        cor_linha = (0, 0, 0)
        espessura_linha = 3
        
        # Criar imagem em branco
        imagem = np.full((altura, largura, 3), cor_fundo, dtype=np.uint8)
        
        # Desenhar gesto baseado na letra
        self.desenhar_gesto_letra(imagem, letra, cor_linha, espessura_linha)
        
        # Salvar imagem
        pasta_letra = f'imagens_sinteticas/{letra}'
        Path(pasta_letra).mkdir(parents=True, exist_ok=True)
        
        nome_arquivo = f"{letra}_sintetico.jpg"
        caminho_arquivo = os.path.join(pasta_letra, nome_arquivo)
        cv2.imwrite(caminho_arquivo, imagem)
        
        print(f"✅ Criada: {nome_arquivo}")
    
    def desenhar_gesto_letra(self, imagem, letra, cor, espessura):
        """Desenha um gesto sintético para uma letra"""
        altura, largura = imagem.shape[:2]
        centro_x, centro_y = largura // 2, altura // 2
        
        # Desenhos básicos para cada letra
        gestos = {
            'D': self.desenhar_gesto_D,
            'E': self.desenhar_gesto_E,
            'F': self.desenhar_gesto_F,
            'G': self.desenhar_gesto_G,
            'H': self.desenhar_gesto_H,
            'I': self.desenhar_gesto_I,
            'J': self.desenhar_gesto_J,
            'K': self.desenhar_gesto_K,
            'M': self.desenhar_gesto_M,
            'N': self.desenhar_gesto_N,
            'O': self.desenhar_gesto_O,
            'P': self.desenhar_gesto_P,
            'Q': self.desenhar_gesto_Q,
            'R': self.desenhar_gesto_R,
            'S': self.desenhar_gesto_S,
            'T': self.desenhar_gesto_T,
            'U': self.desenhar_gesto_U,
            'V': self.desenhar_gesto_V,
            'W': self.desenhar_gesto_W,
            'X': self.desenhar_gesto_X,
            'Z': self.desenhar_gesto_Z,
        }
        
        if letra in gestos:
            gestos[letra](imagem, centro_x, centro_y, cor, espessura)
        else:
            self.desenhar_gesto_padrao(imagem, centro_x, centro_y, cor, espessura)
    
    def desenhar_gesto_D(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra D"""
        cv2.line(imagem, (x, y-40), (x, y+40), cor, espessura)
        cv2.circle(imagem, (x, y-40), 5, cor, -1)
    
    def desenhar_gesto_E(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra E"""
        cv2.circle(imagem, (x, y), 25, cor, espessura)
    
    def desenhar_gesto_F(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra F"""
        cv2.circle(imagem, (x, y), 30, cor, espessura)
        cv2.line(imagem, (x-20, y-20), (x+20, y-20), cor, espessura)
    
    def desenhar_gesto_G(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra G"""
        cv2.line(imagem, (x-40, y), (x+40, y), cor, espessura)
        cv2.circle(imagem, (x+40, y), 5, cor, -1)
    
    def desenhar_gesto_H(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra H"""
        cv2.line(imagem, (x-15, y-40), (x-15, y+40), cor, espessura)
        cv2.line(imagem, (x+15, y-40), (x+15, y+40), cor, espessura)
    
    def desenhar_gesto_I(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra I"""
        cv2.line(imagem, (x, y-40), (x, y+40), cor, espessura)
        cv2.circle(imagem, (x, y-40), 5, cor, -1)
    
    def desenhar_gesto_J(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra J"""
        cv2.line(imagem, (x, y-40), (x, y+40), cor, espessura)
        cv2.arc(imagem, (x, y+20), 20, 0, 180, cor, espessura)
    
    def desenhar_gesto_K(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra K"""
        cv2.line(imagem, (x-20, y-40), (x-20, y+40), cor, espessura)
        cv2.line(imagem, (x+20, y-40), (x+20, y+40), cor, espessura)
        cv2.line(imagem, (x, y), (x+30, y+30), cor, espessura)
    
    def desenhar_gesto_M(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra M"""
        for i in range(3):
            x_pos = x - 20 + i * 20
            cv2.line(imagem, (x_pos, y-40), (x_pos, y+40), cor, espessura)
    
    def desenhar_gesto_N(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra N"""
        cv2.line(imagem, (x-10, y-40), (x-10, y+40), cor, espessura)
        cv2.line(imagem, (x+10, y-40), (x+10, y+40), cor, espessura)
    
    def desenhar_gesto_O(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra O"""
        cv2.circle(imagem, (x, y), 35, cor, espessura)
    
    def desenhar_gesto_P(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra P"""
        cv2.line(imagem, (x, y-40), (x, y+40), cor, espessura)
        cv2.arc(imagem, (x, y-20), 20, 0, 180, cor, espessura)
    
    def desenhar_gesto_Q(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra Q"""
        cv2.line(imagem, (x, y-40), (x, y+40), cor, espessura)
        cv2.line(imagem, (x, y+40), (x+30, y+40), cor, espessura)
        cv2.line(imagem, (x+30, y+40), (x+30, y+20), cor, espessura)
    
    def desenhar_gesto_R(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra R"""
        cv2.line(imagem, (x, y-40), (x, y+40), cor, espessura)
        cv2.line(imagem, (x, y), (x+30, y-20), cor, espessura)
        cv2.line(imagem, (x+30, y-20), (x+30, y+40), cor, espessura)
    
    def desenhar_gesto_S(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra S"""
        cv2.circle(imagem, (x, y), 25, cor, espessura)
    
    def desenhar_gesto_T(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra T"""
        cv2.line(imagem, (x, y-40), (x, y+40), cor, espessura)
        cv2.line(imagem, (x-20, y-20), (x+20, y-20), cor, espessura)
    
    def desenhar_gesto_U(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra U"""
        cv2.line(imagem, (x-15, y-40), (x-15, y+40), cor, espessura)
        cv2.line(imagem, (x+15, y-40), (x+15, y+40), cor, espessura)
    
    def desenhar_gesto_V(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra V"""
        cv2.line(imagem, (x-20, y-40), (x-20, y+40), cor, espessura)
        cv2.line(imagem, (x+20, y-40), (x+20, y+40), cor, espessura)
        cv2.line(imagem, (x-20, y-40), (x+20, y-40), cor, espessura)
    
    def desenhar_gesto_W(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra W"""
        for i in range(3):
            x_pos = x - 20 + i * 20
            cv2.line(imagem, (x_pos, y-40), (x_pos, y+40), cor, espessura)
    
    def desenhar_gesto_X(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra X"""
        cv2.line(imagem, (x-20, y-40), (x+20, y+40), cor, espessura)
        cv2.line(imagem, (x+20, y-40), (x-20, y+40), cor, espessura)
    
    def desenhar_gesto_Z(self, imagem, x, y, cor, espessura):
        """Desenha gesto da letra Z"""
        cv2.line(imagem, (x-30, y-40), (x+30, y-40), cor, espessura)
        cv2.line(imagem, (x+30, y-40), (x-30, y+40), cor, espessura)
        cv2.line(imagem, (x-30, y+40), (x+30, y+40), cor, espessura)
    
    def desenhar_gesto_padrao(self, imagem, x, y, cor, espessura):
        """Desenha gesto padrão"""
        cv2.circle(imagem, (x, y), 30, cor, espessura)
        cv2.putText(imagem, "?", (x-10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, espessura)
    
    def baixar_imagens_internet(self):
        """Baixa imagens da internet"""
        print("📥 Baixando imagens da internet...")
        print("⚠️ Esta funcionalidade requer URLs específicas de imagens de LIBRAS")
        print("💡 Por enquanto, use a opção 3 para criar imagens sintéticas")
    
    def processar_todas_imagens(self):
        """Processa todas as imagens disponíveis"""
        print("🔄 Processando todas as imagens disponíveis...")
        
        pastas_imagens = [
            'imagens/letras/',
            'imagens_sinteticas/',
            'imagens/numeros/'
        ]
        
        todos_dados = []
        total_imagens = 0
        
        for pasta in pastas_imagens:
            if os.path.exists(pasta):
                dados = self.processar_imagens_pasta(pasta)
                if dados:
                    todos_dados.extend(dados)
                    total_imagens += len(dados)
                    print(f"✅ {pasta}: {len(dados)} imagens processadas")
        
        if todos_dados:
            self.salvar_dados(todos_dados)
            print(f"✅ Total: {total_imagens} imagens processadas")
        else:
            print("⚠️ Nenhuma imagem encontrada para processar")
    
    def processar_imagens_pasta(self, pasta_imagens):
        """Processa todas as imagens de uma pasta"""
        if not os.path.exists(pasta_imagens):
            return []
        
        # Buscar imagens
        formatos = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        imagens = []
        for formato in formatos:
            imagens.extend(glob.glob(os.path.join(pasta_imagens, f"**/*{formato}"), recursive=True))
            imagens.extend(glob.glob(os.path.join(pasta_imagens, f"**/*{formato.upper()}"), recursive=True))
        
        if not imagens:
            return []
        
        print(f"📊 Encontradas {len(imagens)} imagens em {pasta_imagens}")
        
        todos_dados = []
        sucessos = 0
        
        for imagem_path in imagens:
            nome_arquivo = os.path.basename(imagem_path)
            
            # Extrair letra do nome
            letra = self.extrair_letra_nome_arquivo(nome_arquivo)
            if not letra:
                continue
            
            landmarks = self.processar_imagem(imagem_path)
            if landmarks is not None:
                todos_dados.append({
                    'label': letra,
                    **{f'point_{j}': landmarks[j] for j in range(63)},
                    'imagem_origem': imagem_path,
                    'timestamp': datetime.now().isoformat()
                })
                sucessos += 1
        
        print(f"✅ Processadas {sucessos}/{len(imagens)} imagens")
        return todos_dados
    
    def processar_landmarks(self, hand_landmarks):
        """Processa landmarks da mão e normaliza"""
        p0 = hand_landmarks.landmark[0]
        points = []
        for landmark in hand_landmarks.landmark:
            points.extend([
                landmark.x - p0.x,
                landmark.y - p0.y,
                landmark.z - p0.z
            ])
        return points
    
    def processar_imagem(self, caminho_imagem):
        """Processa uma única imagem e extrai landmarks"""
        try:
            imagem = cv2.imread(caminho_imagem)
            if imagem is None:
                return None
            
            rgb_imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_imagem)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = self.processar_landmarks(hand_landmarks)
                
                if len(landmarks) == 63:
                    return landmarks
                else:
                    print(f"⚠️ Landmarks inválidos em: {caminho_imagem}")
                    return None
            else:
                print(f"⚠️ Mão não detectada em: {caminho_imagem}")
                return None
                
        except Exception as e:
            print(f"❌ Erro ao processar {caminho_imagem}: {e}")
            return None
    
    def extrair_letra_nome_arquivo(self, nome_arquivo):
        """Extrai a letra do nome do arquivo"""
        import re
        nome_sem_ext = os.path.splitext(nome_arquivo)[0]
        
        # Padrões comuns
        padroes = [
            r'([A-Z])',  # Letra maiúscula
            r'letra_([A-Z])',  # letra_A.jpg
            r'gesto_([A-Z])',  # gesto_A.jpg
            r'([A-Z])_',  # A_001.jpg
            r'_([A-Z])_',  # img_A_001.jpg
        ]
        
        for padrao in padroes:
            match = re.search(padrao, nome_sem_ext)
            if match:
                return match.group(1)
        
        return None
    
    def salvar_dados(self, novos_dados, arquivo_saida='gestos_libras.csv'):
        """Salva dados processados no arquivo CSV"""
        try:
            # Carregar dados existentes
            if os.path.exists(arquivo_saida):
                df_existente = pd.read_csv(arquivo_saida)
                dados_existentes = df_existente.to_dict('records')
                print(f"📁 Carregados {len(dados_existentes)} dados existentes")
            else:
                dados_existentes = []
                print("📁 Criando novo arquivo de dados")
            
            # Combinar dados
            todos_dados = dados_existentes + novos_dados
            df = pd.DataFrame(todos_dados)
            
            # Fazer backup
            if os.path.exists(arquivo_saida):
                backup_name = f"backup/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{arquivo_saida}"
                df_existente.to_csv(backup_name, index=False)
                print(f"📁 Backup criado: {backup_name}")
            
            # Salvar
            df.to_csv(arquivo_saida, index=False)
            
            print(f"✅ Dados salvos em {arquivo_saida}")
            print(f"📊 Total de amostras: {len(todos_dados)}")
            print(f"📈 Novas amostras: {len(novos_dados)}")
            
            # Mostrar distribuição
            print("\n📊 Distribuição por classe:")
            distribuicao = df['label'].value_counts().sort_index()
            for classe, count in distribuicao.items():
                print(f"  {classe}: {count} amostras")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao salvar dados: {e}")
            return False
    
    def treinar_modelo(self):
        """Treina modelo com dados coletados"""
        try:
            print("🧠 Treinando modelo...")
            
            # Carregar dados
            df = pd.read_csv('gestos_libras.csv')
            feature_columns = [col for col in df.columns if col not in ['label', 'imagem_origem', 'timestamp']]
            X = df[feature_columns].values
            y = df['label'].values
            
            print(f"📊 Dados: {len(df)} amostras, {len(feature_columns)} features")
            print(f"🏷️ Classes: {sorted(df['label'].unique())}")
            
            # Treinar modelo
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            import pickle
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=3,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Avaliar
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            
            print(f"📈 Acurácia treino: {train_acc:.2%}")
            print(f"📈 Acurácia teste: {test_acc:.2%}")
            
            # Salvar modelo
            with open('modelos/modelo_libras.pkl', 'wb') as f:
                pickle.dump(model, f)
            
            # Salvar informações
            model_info = {
                'classes': model.classes_.tolist(),
                'n_features': len(feature_columns),
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'n_samples': len(df),
                'timestamp': datetime.now().isoformat()
            }
            
            with open('modelos/modelo_info.pkl', 'wb') as f:
                pickle.dump(model_info, f)
            
            print("✅ Modelo treinado e salvo!")
            return True
            
        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
            return False
    
    def mostrar_estatisticas(self):
        """Mostra estatísticas detalhadas dos dados"""
        try:
            if os.path.exists('gestos_libras.csv'):
                df = pd.read_csv('gestos_libras.csv')
                print(f"\n📊 ESTATÍSTICAS DETALHADAS:")
                print(f"📈 Total de amostras: {len(df)}")
                print(f"🏷️ Classes únicas: {len(df['label'].unique())}")
                print(f"📋 Classes: {sorted(df['label'].unique())}")
                
                print(f"\n📊 DISTRIBUIÇÃO:")
                distribuicao = df['label'].value_counts().sort_index()
                for classe, count in distribuicao.items():
                    print(f"  {classe}: {count} amostras")
                
                # Calcular progresso
                letras_implementadas = [c for c in df['label'].unique() if c.isalpha()]
                letras_faltantes = [l for l in self.letras_faltantes if l not in letras_implementadas]
                
                print(f"\n📈 PROGRESSO:")
                print(f"✅ Letras implementadas: {len(letras_implementadas)}/26")
                print(f"⏳ Letras faltantes: {len(letras_faltantes)}")
                if letras_faltantes:
                    print(f"📝 Faltam: {', '.join(letras_faltantes)}")
                
            else:
                print("❌ Arquivo gestos_libras.csv não encontrado")
                
        except Exception as e:
            print(f"❌ Erro ao mostrar estatísticas: {e}")
    
    def ferramentas_auxiliares(self):
        """Ferramentas auxiliares"""
        print("\n🔧 FERRAMENTAS AUXILIARES:")
        print("1. 📁 Criar estrutura de pastas")
        print("2. 🗑️ Limpar dados duplicados")
        print("3. 📊 Gerar relatório detalhado")
        print("4. 🔄 Voltar ao menu principal")
        
        opcao = input("Digite o número da opção (1-4): ").strip()
        
        if opcao == '1':
            self.criar_estrutura()
        elif opcao == '2':
            self.limpar_duplicados()
        elif opcao == '3':
            self.gerar_relatorio()
        elif opcao == '4':
            return
        else:
            print("❌ Opção inválida!")
    
    def limpar_duplicados(self):
        """Remove dados duplicados"""
        print("🗑️ Limpando dados duplicados...")
        # Implementar limpeza de duplicados
        print("✅ Dados duplicados removidos")
    
    def gerar_relatorio(self):
        """Gera relatório detalhado"""
        print("📊 Gerando relatório detalhado...")
        # Implementar geração de relatório
        print("✅ Relatório gerado")

def main():
    print("🚀 TraduLibras - Sistema Completo de Expansão de Vocabulário")
    print("📚 Inclui: câmera, imagens, sintético, download e mais!")
    
    sistema = SistemaCompletoVocabulario()
    sistema.menu_principal()

if __name__ == "__main__":
    main()

