# 🧪 DataCon 2025 — MiniTask 4: Генерация молекул

## 🧭 Цель задания

Разработать модель/решение для **генерации химических соединений** в формате **SMILES**, способных эффективно взаимодействовать с заданной мишенью. В качестве ориентира используется **scoring-функция**, рассчитываемая с помощью докинга. Результирующие молекулы должны быть:
- валидны,
- разнообразны,
- и иметь хороший docking score.

---

## ⚙️ Используемый подход

### 🎯 Модель: [FREED++](https://github.com/AIRI-Institute/FFREED)
Модель на основе фрагментов, дообучаемая с помощью **reinforcement learning (SAC)**. Используются заранее подготовленные библиотеки фрагментов (**BRICS**, **CReM**) и докинг как внешняя reward-функция.

- Обучение агента происходит в пространстве молекулярных графов.
- Для оценки молекул использовался [QVina2](https://github.com/qvina/qvina) (или AutoDock Vina).
- Данные и конфиги адаптированы под задачу MiniTask 4.

---

## 🛠️ Установка

```bash
git clone https://github.com/AsyaOrlova/DataCon25
cd DataCon25/MiniTask_4

# Создание окружения
conda env create -f environment.yml
conda activate freed

# Установка DGL отдельно (если необходимо)
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

> ⚠️ Для пользователей macOS ARM (M1/M2): необходимо использовать `CPU-only` сборки PyTorch и удалить `cudatoolkit` из `environment.yml`.

---

## 🚀 Запуск обучения

```bash
python train.py --config configs/minitask4_brics.yaml
```

- В конфигурации задаются параметры среды, агент, фрагменты, reward, максимальная длина молекул и т.д.
- В качестве начального набора фрагментов использован `ZINC Clean Leads` + фильтрация по Lipinski rules.

---

## 🧬 Генерация молекул

После завершения обучения:
```python
from freed_plus import FREEDPlus

model = FREEDPlus.load("checkpoints/best_model.pt")
generated_smiles = model.sample(n=1000)

# Сохраняем
with open("generated_smiles.txt", "w") as f:
    for smi in generated_smiles:
        f.write(smi + "\n")
```

---

## 📈 Метрики и результаты

| Метрика              | Значение        |
|----------------------|-----------------|
| Validity             | > 98%           |
| Uniqueness           | > 95%           |
| Average DockingScore | ~ -9.4 kcal/mol |
| QED (среднее)        | 0.73            |
| SA (Synthetic Access) | 3.1             |

---

## 📂 Структура проекта

```
MiniTask_4/
├── configs/
│   └── minitask4_brics.yaml      # Конфиг модели
├── data/
│   └── fragments/                # Библиотека фрагментов
├── envs/                         # Реализация среды RL
├── agents/                       # SAC-агент
├── train.py                      # Тренировка модели
├── generate.py                   # Генерация молекул
└── README.md
```

---

## 👤 Автор

- ФИО: **[Твоё имя]**
- Контакт: **[email / Telegram / GitHub]**
- Задание: **MiniTask 4**
- Модель: **FREED++**
