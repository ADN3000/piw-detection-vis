Claro. Abaixo está a parte geral em **Markdown puro**, pronta para você copiar e colar no `historico_projeto.md`.

```markdown
# Histórico Técnico — Projeto DINOv3-SAR

**Projeto:** DINOv3 aplicado à detecção de pessoas em Busca e Salvamento Marítimo  
**Ambiente:** Windows / NVIDIA RTX 4060 Laptop  
**Objetivo:** Registrar decisões de arquitetura, configuração do ambiente, softwares instalados e evolução experimental do projeto.

---

## 1. Concepção do Projeto

O projeto foi concebido para investigar técnicas de Inteligência Artificial e Visão Computacional capazes de auxiliar a detecção de pessoas em ambiente marítimo, com foco em aplicações de Busca e Salvamento (SAR — Search and Rescue) utilizando imagens e vídeos provenientes de sensores EO/IR aerotransportados.

O problema operacional não é tratado simplesmente como a detecção genérica da classe `person`.

O cenário considerado envolve condições como:

- pessoas representadas por poucos pixels;
- grandes distâncias entre sensor e alvo;
- movimento da aeronave;
- movimento do sensor/gimbal;
- movimento da pessoa;
- ondas e espuma;
- reflexos solares;
- baixo contraste;
- oclusões temporárias;
- diferentes estados do mar;
- diferentes condições de iluminação;
- necessidade de processamento com baixa latência;
- necessidade de limitar falsos alarmes apresentados ao operador.

A arquitetura deverá permitir futuramente uma operação do tipo:

Sensor EO/IR  
↓  
Vídeo  
↓  
Sistema de IA  
↓  
Detecção  
↓  
Tracking  
↓  
Alerta  
↓  
Operador  
↓  
Confirmação

A Inteligência Artificial é considerada inicialmente uma ferramenta de apoio à detecção e à consciência situacional, e não necessariamente um substituto do operador.

---

## 2. Estratégia Geral

O desenvolvimento será dividido conceitualmente entre treinamento e inferência operacional.

### Treinamento

O treinamento poderá ser realizado utilizando imagens anotadas.

Fluxo:

Imagem  
+  
Bounding Box / Máscara  
↓  
Treinamento  
↓  
Modelo treinado

Não é necessário que o treinamento inicial seja realizado diretamente sobre vídeos.

### Inferência operacional

O modelo treinado será posteriormente utilizado sobre vídeo.

Fluxo:

Vídeo EO/IR  
↓  
Captura do frame  
↓  
Pré-processamento  
↓  
Modelo de IA  
↓  
Detecção  
↓  
Bounding Box  
↓  
Tracker  
↓  
Visualização / Alerta

---

## 3. DINOv3

O DINOv3 será inicialmente investigado como backbone e extrator de características visuais.

O modelo recebe uma imagem e produz representações numéricas (embeddings) que descrevem características do conteúdo visual.

Uma das representações de interesse são os `patch tokens`.

Conceitualmente:

Imagem  
↓  
Divisão em patches  
↓  
Vision Transformer  
↓  
Embedding para cada patch

Essas representações poderão ser utilizadas para investigar se regiões contendo água, espuma, embarcações, pessoas e outros objetos apresentam características suficientemente diferentes no espaço de embeddings.

---

## 4. INSID3

O notebook utilizado como ponto de partida emprega o INSID3 associado ao DINOv3.

Fluxo conceitual:

Imagem de referência  
+  
Máscara de referência  
↓  
DINOv3  
↓  
Representações visuais  
↓  
INSID3  
↓  
Segmentação da imagem alvo

O INSID3 será mantido inicialmente como uma das abordagens experimentais.

Entretanto, o projeto não será arquiteturalmente dependente do INSID3.

---

## 5. Análise de Embeddings

O notebook inicial contém experimentos de extração de:

- `x_norm_patchtokens`
- `x_norm_clstoken`

Também foram identificados experimentos utilizando técnicas como:

- K-Means;
- Hierarchical Clustering.

O objetivo dessa linha experimental será compreender como o DINOv3 representa internamente diferentes regiões das imagens.

Uma das questões a serem investigadas é se regiões contendo pessoas apresentam embeddings distinguíveis do fundo marítimo, ondas, espuma e outros elementos presentes no ambiente operacional.

---

## 6. Processamento de Vídeo

O sistema deverá futuramente aceitar diferentes fontes de vídeo, incluindo:

- arquivo de vídeo;
- câmera;
- stream de vídeo;
- RTSP;
- futuramente, entrada proveniente de sensor EO/IR.

Pipeline básico:

Vídeo  
↓  
OpenCV  
↓  
Frame  
↓  
Pré-processamento  
↓  
Modelo  
↓  
Detecção  
↓  
Pós-processamento  
↓  
Bounding Box  
↓  
Visualização

Deverão ser registrados indicadores como:

- FPS;
- latência por frame;
- recall;
- precision;
- falsos positivos;
- falsos negativos;
- estabilidade temporal das detecções.

Para a aplicação SAR, atenção especial será dada ao recall, sem ignorar a quantidade de falsos alarmes apresentada ao operador.

---

## 7. Tracking

Está prevista a integração futura de um módulo de tracking.

O detector identifica objetos individualmente em cada frame.

Exemplo:

Frame 100 → Pessoa  
Frame 101 → Pessoa  
Frame 102 → Pessoa

O tracker procura associar essas detecções ao mesmo objeto:

Frame 100 → Pessoa ID 7  
Frame 101 → Pessoa ID 7  
Frame 102 → Pessoa ID 7

Isso poderá ser particularmente importante em ambiente marítimo devido às oclusões temporárias provocadas pelas ondas.

Exemplo:

Frame 100 → detectado  
Frame 101 → detectado  
Frame 102 → não detectado  
Frame 103 → detectado  
Frame 104 → detectado

A informação temporal poderá futuramente ser investigada como mecanismo para reduzir falsos alarmes e aumentar a confiabilidade de um alerta apresentado ao operador.

Uma das opções inicialmente consideradas para tracking é o ByteTrack.

---

## 8. Estrutura Geral da Pesquisa

Diretório principal:

C:\Pesquisa\Mestrado\

Estrutura planejada:

Mestrado/
├── projetos/
├── datasets/
├── checkpoints/
├── papers/
├── resultados/
└── docs/

Essa separação permite reutilizar datasets e checkpoints em diferentes experimentos sem duplicação desnecessária.

---

## 9. Projeto Principal

Diretório atual:

C:\Pesquisa\Mestrado\projetos\dino3-sar

Estrutura planejada:

dino3-sar/
│
├── .venv/
├── configs/
├── notebooks/
├── experiments/
│
├── src/
│   ├── video/
│   ├── preprocessing/
│   ├── models/
│   ├── inference/
│   ├── embeddings/
│   ├── tracking/
│   ├── visualization/
│   ├── evaluation/
│   └── utils/
│
├── outputs/
│   ├── images/
│   ├── videos/
│   ├── embeddings/
│   └── logs/
│
├── tests/
│
├── train.py
├── infer_image.py
├── infer_video.py
├── infer_camera.py
├── requirements.txt
├── README.md
└── .gitignore

---

## 10. Ambiente Python

Inicialmente foi identificado:

Python 3.14.13

Por ser uma versão muito recente e potencialmente problemática para compatibilidade com bibliotecas científicas, foi decidido não utilizá-la no ambiente do projeto.

O Python 3.14 foi mantido instalado no computador.

Foi instalado adicionalmente:

Python 3.11.9

O Python 3.11 foi escolhido para aumentar a compatibilidade com o ecossistema PyTorch, DINOv3 e INSID3.

---

## 11. Ambiente Virtual

Foi criado um ambiente virtual exclusivo para o projeto:

C:\Pesquisa\Mestrado\projetos\dino3-sar\.venv

Comando utilizado:

py -3.11 -m venv .venv

Ativação:

.\.venv\Scripts\Activate.ps1

O executável Python utilizado pelo projeto foi confirmado como:

C:\Pesquisa\Mestrado\projetos\dino3-sar\.venv\Scripts\python.exe

Portanto, as bibliotecas instaladas para a pesquisa permanecem isoladas das demais instalações Python do computador.

---

## 12. Git

Versão identificada:

Git 2.53.0.windows.2

O Git será utilizado para controle de versão do código e registro da evolução dos experimentos.

Datasets, checkpoints grandes, vídeos e outros artefatos de grande volume não deverão ser armazenados diretamente no repositório Git.

---

## 13. Hardware NVIDIA

GPU identificada:

NVIDIA GeForce RTX 4060 Laptop GPU

Memória de vídeo:

8188 MiB (~8 GB VRAM)

Driver NVIDIA:

591.59

O `nvidia-smi` informou suporte do driver a CUDA 13.1.

Foi decidido não instalar manualmente o CUDA Toolkit neste momento, pois a distribuição utilizada do PyTorch contém os componentes CUDA necessários para execução na GPU.

---

## 14. PyTorch e CUDA

Foi instalado o PyTorch dentro do ambiente virtual do projeto.

Versões instaladas:

- PyTorch: 2.13.0+cu130
- TorchVision: 0.28.0+cu130
- TorchAudio: 2.11.0+cu130
- CUDA utilizada pelo PyTorch: 13.0

Também foram instaladas automaticamente dependências como:

- NumPy;
- Pillow;
- NetworkX;
- Jinja2;
- FSSpec;
- SymPy;
- FileLock;
- Typing Extensions.

---

## 15. Validação da GPU

Foi realizado teste através do PyTorch.

Resultado:

PyTorch: 2.13.0+cu130  
CUDA PyTorch: 13.0  
CUDA disponível: True  
GPU: NVIDIA GeForce RTX 4060 Laptop GPU

Portanto, foi confirmado o funcionamento do seguinte caminho:

Python 3.11  
↓  
PyTorch  
↓  
CUDA  
↓  
NVIDIA GeForce RTX 4060 Laptop GPU

Status: GPU disponível para processamento utilizando PyTorch.

---

## 16. Estado Atual do Ambiente

Componentes configurados:

- Windows: OK
- Git: OK
- Python 3.11.9: OK
- Ambiente virtual `.venv`: OK
- Driver NVIDIA: OK
- RTX 4060: OK
- PyTorch: OK
- CUDA no PyTorch: OK
- GPU reconhecida pelo PyTorch: OK

Ainda não configurados nesta etapa:

- DINOv3;
- INSID3;
- OpenCV;
- ByteTrack;
- demais dependências específicas do pipeline.

---

## 17. Próximas Etapas

Sequência inicialmente planejada:

Ambiente base  
↓  
DINOv3  
↓  
Teste DINOv3 + RTX 4060  
↓  
Extração de embeddings  
↓  
INSID3  
↓  
Reprodução do notebook inicial  
↓  
Inferência em imagem  
↓  
Inferência em vídeo  
↓  
Detecção  
↓  
Tracking  
↓  
Avaliação científica e operacional

---

## 18. Diário de Desenvolvimento

A partir deste ponto, alterações relevantes no ambiente e no projeto deverão ser registradas cronologicamente.

### 2026-08-03 — Configuração inicial

- Estrutura inicial do projeto definida.
- Python 3.11.9 instalado.
- Ambiente virtual `.venv` criado.
- Git 2.53.0 identificado.
- NVIDIA GeForce RTX 4060 Laptop GPU identificada.
- Driver NVIDIA 591.59 identificado.
- PyTorch 2.13.0+cu130 instalado.
- TorchVision 0.28.0+cu130 instalado.
- TorchAudio 2.11.0+cu130 instalado.
- CUDA reconhecida pelo PyTorch.
- `torch.cuda.is_available()` retornou `True`.
- GPU NVIDIA GeForce RTX 4060 Laptop GPU reconhecida pelo PyTorch.

### Próxima atividade

Instalação e validação do DINOv3 no ambiente virtual do projeto.
```
### 2026-08-03 — Instalação do DINOv3

- Repositório oficial do DINOv3 clonado em `external/dinov3`.
- DINOv3 instalado em modo editável no ambiente virtual.
- Versão instalada: DINOv3 0.0.1.
- Importação do módulo `dinov3` validada com sucesso.
- PyTorch permaneceu na versão 2.13.0+cu130.
- CUDA 13.0 permaneceu disponível no PyTorch.
- `torch.cuda.is_available()` retornou `True`.
- NVIDIA GeForce RTX 4060 Laptop GPU reconhecida após a instalação do DINOv3.
- Ambiente DINOv3 + PyTorch + CUDA + GPU validado com sucesso.

### Próxima atividade

Carregamento do primeiro backbone DINOv3 e execução de inferência utilizando a RTX 4060.