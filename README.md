# Verificação Facial e Avaliação de Robustez contra Deepfakes

Pipeline completo de **verificação facial** e **avaliação de vulnerabilidade a ataques de face swap (deepfakes)** utilizando:

- **YOLOv8-Face** — detecção de faces + extração de 5 landmarks para alinhamento
- **AdaFace IR-50** — extração de embeddings faciais (512-d) pré-treinado no WebFace4M
- **FaceFusion (inswapper_128)** — geração automatizada de face swaps (deepfakes)
- **Similaridade Cosseno** — métrica de verificação biométrica

## Objetivo

Avaliar a **robustez do sistema biométrico** (YOLOv8 + AdaFace) contra ataques de **deepfake por face swap**, medindo a taxa com que imagens manipuladas conseguem enganar o classificador facial. Os experimentos são conduzidos em dois cenários:

1. **Dataset Pessoal** — imagens controladas de pessoas conhecidas
2. **LFW (Labeled Faces in the Wild)** — benchmark acadêmico padrão com 13.233 imagens

---

## Requisitos

- Docker instalado (versão com suporte a `docker-compose`)
- GPU NVIDIA + drivers CUDA
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Como Inicializar o Ambiente (Docker)

Todo o projeto roda dentro de um container Docker configurado com a biblioteca CUDA, garantindo acesso direto à sua placa de vídeo.

```bash
# 1. No terminal do seu servidor/computador, suba o container:
sudo docker-compose -f docker-compose.gpu.yml up -d

# 2. Agora "entre" no terminal do container para poder executar os Python scripts:
sudo docker exec -it ia_gpu bash

# 🪄 Pronto! Seu terminal vai mudar para algo parecido com "root@a2bd43...:/app#". 
# Todos os comandos "python app/nome_do_arquivo.py" a seguir devem ser digitados AQUI dentro!
```

> **Dica:** Se aparecer um erro no passo 1 dizendo que "o container ia_gpu já existe", force a remoção com `sudo docker rm -f ia_gpu` e tente de novo.

---

## Estrutura do Projeto & Como Rodar Cada Script

Para evitar confusão, os scripts estão organizados em **Núcleo Acadêmico (Paper)** e **Utilitários Extras**.

### 📚 1. Núcleo Acadêmico do PAPER (pasta `app/`)
Estes são os arquivos centrais responsáveis por gerar os dados da sua pesquisa LFW.

| O que ele faz? | Como rodar no terminal do Docker? |
|---|---|
| **`app/baseline_lfw.py`**<br>Avalia 6000 pares reais do protocolo LFW. Extrai a acurácia, FAR, e FRR "originais" da IA para usarmos de base. É a métrica do mundo honesto. | `python app/baseline_lfw.py` |
| **`app/run_swaps_lfw.py`**<br>**(O Ataque Principal do Artigo)**. Gera automaticamente 100 deepfakes usando pessoas do LFW, injeta contra a IA, calcula Testes Estatísticos (IC 95%) e salva os gráficos. | `python app/run_swaps_lfw.py` |
| **`app/generate_paper_tables.py`**<br>Junta tudo que foi gerado pelos scripts acima e constrói as tabelas formatadas de Resumo e Consolidação que vão direto pro Word do artigo. | `python app/generate_paper_tables.py` |
| **`app/api.py`**<br>Transforma o sistema inteiro num serviço de internet "FastAPI". Se você quiser conectar essa biometria em um aplicativo real futuramente, é ele a porta de entrada. | `uvicorn app.api:app --host 0.0.0.0 --port 8000` |
| **`app/net.py`**<br>É onde o cérebro matemático "Backbone IResNet-50" (do AdaFace) fica instalado. Você não roda esse arquivo, os outros scripts o chamam. | *(Apenas importado pelos outros)* |

<br>

### 🛠️ 2. Utilitários Livres & Testes com Amigos (pasta `app/utils/`)
Esses arquivos fazem a mesmíssima coisa, só que aplicados nas fotos soltas da sua equipe e dataset pessoal, ótimos para tirar "prova real" validada.

| O que ele faz? | Como rodar no terminal do Docker? |
|---|---|
| **`app/run_swaps_experiment.py`**<br>Cruza as pessoas do seu arquivo `data/dataset_pessoal/originais`, arranca o rosto de um, cola no corpo do outro e joga contra a IA. | `python app/run_swaps_experiment.py` |
| **`app/utils/compare_faces.py`**<br>Pega duas imagens quaisquer e diz se a IA acha a pessoa igual ou não. | `python app/utils/compare_faces.py data/rostoA.png data/rostoB.png` |
| **`app/utils/swap_faces.py`**<br>Faz manualmente 1 único deepfake e salva onde você mandar. Bom para criar imagens engraçadas fora da base de testes. | `python app/utils/swap_faces.py data/rosto.jpg data/corpo_alvo.png resultado.jpg` |
| **`app/utils/run_experiment.py`**<br>Lê toooodas as fotos reais da sua equipe e compara "todo mundo contra todo mundo" e gera uma tabela Top 15. | `python app/utils/run_experiment.py` |
| **`app/utils/visualize_alignment.py`**<br>Um utilitário visual que gera uma imagem em PDF demonstrando os 5 pontinhos verdes (Nariz, Boca e Olho) que a YOLO enxerga. Excelente ferramenta pra colocar no artigo como contexto teórico! | `python app/utils/visualize_alignment.py` |
| **`app/utils/test_gpu.py`**<br>Script rápido pra validar se o Docker de fato está sugando as informações da sua placa de vídeo NVIDIA. | `python app/utils/test_gpu.py` |

---

## Pipeline de Avaliação

```
                    ┌─────────────────────────────────────────┐
                    │         ETAPA 1: GERAÇÃO DE FRAUDE      │
                    │                                         │
  Foto Rosto A ───► │   FaceFusion (inswapper_128)            │ ──► Imagem Swap
  Foto Corpo B ───► │   Cola rosto A no corpo de B            │     (Deepfake)
                    └─────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │       ETAPA 2: AUDITORIA BIOMÉTRICA     │
                    │                                         │
   Imagem Swap ───► │  YOLOv8-Face → Crop → AdaFace → Emb.   │ ──► Similaridade
  Foto Original ──► │  YOLOv8-Face → Crop → AdaFace → Emb.   │     Cosseno
                    └─────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │         ETAPA 3: VEREDICTO              │
                    │                                         │
                    │  sim >= τ  →  🚨 Deepfake ENGANOU a IA  │
                    │  sim <  τ  →  🛡️ Deepfake foi BARRADO   │
                    └─────────────────────────────────────────┘
```

---

## Resultados Principais (Validados no Artigo)

Os experimentos foram avaliados em 3 regimes distintos de complexidade. O threshold de aceitação adotado como referência de segurança média foi $\tau = 0.45$.

### 1. Imagens Estáticas (LFW)
- **Baseline**: 6.000 pares (Acurácia = 95.97%, EER = 6.60%)
- **Tamanho do Ataque**: 3.000 face swaps gerados
- **Attack Success Rate (ASR)**: **92.8%** 
- **Score Shift ($\Delta s$)**: **0.6937** (Deslocamento estatístico massivo para a região de aceitação)

### 2. Vídeos Não-Restritos (YouTube Faces - YTF)
- **Baseline**: 5.000 pares de vídeo (Acurácia = 76.74%, EER = 23.56%)
- **Tamanho do Ataque**: 999 vídeos manipulados (agregando 5 frames/vídeo)
- **Attack Success Rate (ASR)**: **34.3%**
- **Score Shift ($\Delta s$)**: **0.1353** (Transferência de identidade mitigada pela compressão temporal e variação de pose)

### 3. Injeção em Tempo Real (Câmera Virtual v4l2loopback)
- **Teste de Carga**: 2654 frames processados (Duração de ~109.8 segundos)
- **Performance do Pipeline**: **28.6 FPS** (Viabilidade operacional confirmada via GPU)
- **Viabilidade da Injeção**: Demonstrado com sucesso o *bypass* do hardware físico através da injeção contínua do vídeo manipulado (*face swap* gerado pelo FaceFusion) diretamente no dispositivo de câmera virtual. Isso prova o potencial técnico para ataques de injeção direta em softwares de videoconferência.

---

## Arquivos de Resultados

| Arquivo | Descrição |
|---------|-----------|
| `app/results/paper_plots/summary_metrics.csv` | Tabela consolidada com FAR, FRR, ASR e EER de todos os cenários |
| `app/results/lfw_scores.csv` | Scores cosseno dos 6.000 pares do baseline LFW |
| `app/results/lfw_swap_attack_results.csv` | Scores detalhados de cada um dos 3.000 swaps LFW |
| `app/results/ytf_scores.csv` | Scores cosseno dos 5.000 pares de vídeo do YTF |
| `app/results/ytf_swap_attack_results.csv` | Scores detalhados de cada um dos 999 vídeos manipulados |
| `app/results/realtime_attack_log.csv` | Log completo da sessão em tempo real contendo framerate e latência frame a frame |
| `app/results/paper_plots/` | Diretório contendo todos os gráficos do artigo (ROC, DET, Boxplots, Histogramas) |
| `data/dataset_pessoal/resultados_dataset_pessoal.csv` | Teste manual e pareamento da equipe local |

---

## Tecnologias

| Componente | Tecnologia | Versão |
|-----------|-----------|--------|
| Container | Docker + NVIDIA CUDA | 12.2.2 + cuDNN 8 |
| Linguagem | Python | 3.13.7 |
| Deep Learning | PyTorch | 2.6.0+cu124 |
| Visão Computacional | OpenCV, Ultralytics, ONNXRuntime-GPU | — |
| Embedding Facial | AdaFace IR-50 (CVLFace) | — |
| Face Swap | FaceFusion (inswapper_128) | 3.x |
| Protocolo Tempo-Real | WebSockets (`websockets` + `asyncio`) | — |
| Injeção / Câmera Virtual | `v4l2loopback` via `pyvirtualcam` | — |

---

## Referências

- **AdaFace**: Kim, M. et al. "AdaFace: Quality Adaptive Margin for Face Recognition." CVPR, 2022.
- **LFW**: Huang, G. B. et al. "Labeled Faces in the Wild." University of Massachusetts, 2007.
- **FaceFusion**: https://github.com/facefusion/facefusion
- **YOLOv8-Face**: https://github.com/akanametov/yolo-face
