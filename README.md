# Classificação de recall em mamografias

Este projeto implementa, em PyTorch, um pipeline reproduzível para classificar
exames mamográficos quanto à necessidade de *recall*. As imagens de cada exame
são processadas individualmente por uma CNN e suas probabilidades são agregadas
por média para produzir uma única predição por `AccessionNumber`.





## 1. Funcionamento do pipeline

O fluxo principal realiza as seguintes etapas:

1. lê os rótulos de `DATA_DIR/train.csv`;
2. relaciona cada pasta de imagens ao CSV pelo `AccessionNumber`;
3. separa desenvolvimento em treino e validação com estratificação e
   agrupamento por `PatientID`;
4. mantém `proc/test/png/test` como conjunto de teste independente;
5. treina a CNN com entropia cruzada ponderada;
6. seleciona o melhor checkpoint pela menor perda de validação;
7. agrega por média as probabilidades das imagens de cada exame;
8. seleciona o limiar somente na validação, quando usada a política de Youden;
9. calcula as métricas finais no teste e, opcionalmente, intervalos de confiança
   por *bootstrap* de pacientes.

As arquiteturas disponíveis no comando `benchmark` são:

- `efficientnet_b0`;
- `densenet121`;


## 2. Requisitos

- Windows com PowerShell;
- Python 3.12;
- espaço em disco para os DICOMs, PNGs, pesos e resultados;
- acesso à internet na primeira execução dos modelos obtidos do Hugging Face;
- GPU NVIDIA opcional. O pipeline funciona em CPU, porém o treinamento completo
  é consideravelmente mais demorado.

## 3. Preparação do ambiente

Abra o PowerShell na pasta do projeto (`E:\tcc`) e execute:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```



Confira a instalação e a disponibilidade de CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponível:', torch.cuda.is_available())"
```



## 4. Organização esperada dos dados

O projeto já está configurado para a seguinte estrutura:

```text
E:\tcc
├── DATA_DIR
│   └── train.csv
├── raw_images_dir
│   ├── train\<AccessionNumber>\*.dcm
│   └── test\<AccessionNumber>\*.dcm
├── proc
│   ├── train\png\train\<AccessionNumber>\*.png
│   └── test\png\test\<AccessionNumber>\*.png
├── artifacts\benchmarks
├── mammo_pipeline.py
└── requirements.txt
```


## 5. Teste rápido do benchmark

```
.\.venv\Scripts\python.exe .\mammo_pipeline.py benchmark `
  --models efficientnet_b0 `
  --epochs 1 `
  --batch-size 2 `
  --workers 0 `
  --device cpu `
  --image-size 224 `
  --max-train-exams 32 `
  --max-validation-exams 16 `
  --max-test-exams 16 `
  --run-id smoke_efficientnet
```



## 6. Reproduzir os experimentos principais

Os scripts abaixo executam as sementes `42`, `123` e `2026`, com imagens
de `384 x 384` pixels, lote 8, cinco épocas máximas, limiar de Youden
selecionado na validação e 1.000 reamostragens *bootstrap* por paciente.

EfficientNet-B0:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_efficientnet384_multiseed.ps1
```

DenseNet-121:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_densenet121_384_multiseed.ps1
```



```powershell
.\.venv\Scripts\python.exe .\mammo_pipeline.py benchmark `
  --models efficientnet_b0 densenet121 `
  --epochs 5 `
  --batch-size 8 `
  --workers 0 `
  --device cuda `
  --seed 42 `
  --image-size 384 `
  --threshold-policy youden `
  --bootstrap-iterations 1000 `
  --learning-rate 2e-5 `
  --weight-decay 1e-4 `
  --patience 2 `
  --min-delta 1e-4 `
  --gradient-clip 1.0 `
  --run-id benchmark_gpu_seed42 `
  --output-root .\artifacts\benchmarks
```





## 7. Converter DICOM para PNG


```powershell
.\.venv\Scripts\python.exe .\mammo_pipeline.py preprocess `
  --raw-root .\raw_images_dir\test `
  --output-root .\proc\generated\png
```

Cada diretório de saída corresponde a um `AccessionNumber`. O nome do PNG usa
os metadados de lateralidade e incidência do DICOM; ocorrências repetidas
recebem sufixos numéricos e são preservadas.
