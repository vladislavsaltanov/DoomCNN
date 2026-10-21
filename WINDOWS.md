# Запуск проекта на Windows

Проект начинался на macOS, обучение S3–S8 шло на Windows 11 с RTX 3060 Ti. Код сам выбирает видеокарту NVIDIA (CUDA), если она есть. Особенности Windows, найденные по ходу (таймер Windows 11, память, кодировка видео), учтены в коде и описаны в `research_log.md`.

## 1. Что поставить

- Windows 10 или 11, 64 бит.
- Свежий драйвер NVIDIA (GeForce Experience или nvidia.com).
- Python 3.12, 64 бит, с python.org. В установщике отметьте «Add python.exe to PATH». Проверка: `py -3.12 --version`.
- Git for Windows (git-scm.com). Он нужен для истории проекта и для Claude Code.
- По желанию Claude Code для Windows, чтобы продолжить работу по плану.

## 2. Перенос

1. Скопируйте `DoomCNN_windows.zip` на Windows (флешка, облако).
2. Распакуйте в папку без пробелов, например `C:\Projects\DoomCNN`. Имена части файлов русские, поэтому распаковывайте через PowerShell (`tar -xf DoomCNN_windows.zip -C C:\Projects`) или 7-Zip, а не через «Извлечь всё» старых версий Windows.
3. Внутри уже есть история git, план и спек (`docs/`), журнал исследования и ledger (`.superpowers/`). Датасет и модели не переносятся: датасет перезаписывается за пару минут, модели обучаются заново.

## 3. Окружение

В PowerShell, в папке проекта:

```powershell
cd C:\Projects\DoomCNN
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
# PyTorch с CUDA: точную команду возьмите на pytorch.org/get-started (Stable, Windows, Pip, CUDA).
# Проверено: torch 2.14.1 со сборкой CUDA 13.2 (сборки cu128 для этой версии нет):
.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu132
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install -e .
```

Проверка видеокарты:

```powershell
.venv\Scripts\python -c "import torch, vizdoom; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Ожидается `True` и имя карты. Если `False`, то стоит PyTorch без CUDA: удалите его (`pip uninstall torch`) и поставьте заново командой с pytorch.org.

`requirements-lock.txt` фиксирует версии, на которых всё проверено на Mac. `torch==2.14.1` в нём совпадает с CUDA-сборкой `2.14.1+cu128`, поэтому pip её не заменит.

## 4. Проверка

```powershell
.venv\Scripts\python -m pytest -q
```

Ожидается `78 passed`. Если падают только тесты с маркером `vizdoom`, значит не запускается ViZDoom. Тогда проверьте отдельно: `.venv\Scripts\python -m pytest -q -m "not vizdoom"` должно дать `63 passed`. Скопируйте ошибку в Claude.

## 5. Повтор S1 и S2 (несколько минут)

Учитель и датасет должны совпасть с Mac. ViZDoom не обещает бит-в-бит одинаковую игру на разных ОС, поэтому небольшие расхождения возможны. Их нужно записать в журнал.

```powershell
.venv\Scripts\python scripts\run_baselines.py
.venv\Scripts\python -m doomagent.record --out data/train --set base
.venv\Scripts\python scripts\s2_report.py data/train
```

С чем сравнивать:

- учитель на Mac: 22.60 [21.73, 23.27], kills/shot 0.923 (`reports/s1_teacher.md`);
- датасет на Mac: 300 эпизодов, 85 064 шага, доли меток 0.270 / 0.258 / 0.173 / 0.158 / 0.141 (`reports/s2_dataset.md`).

`run_baselines.py` и `s2_report.py` перезапишут файлы в `reports/`. Расхождение с Mac будет видно в `git diff`.

## 6. S3: обучение ученика

Семь прогонов из плана (`bc0` и абляции), несколько одновременно:

```powershell
.venv\Scripts\python scripts\s3_runs.py --parallel 3 --workers 4
```

- `--parallel`: сколько прогонов идут сразу. Каждые 2 эпохи ученик играет 30 эпизодов на процессоре, поэтому на многоядерном процессоре 3–4 прогона одновременно выгоднее, чем один.
- `--workers`: процессы загрузки данных на прогон. Если видеокарта простаивает, увеличьте.
- Прогресс: `runs\bc0.out` и другие `.out`. Строка с `val_score` появляется каждые 2 эпохи. Смотреть так: `Get-Content runs\bc0.out -Wait`.
- Очередь можно прервать (Ctrl+C) и запустить снова: законченные прогоны пропускаются, незаконченные начинаются заново.

После обучения:

```powershell
.venv\Scripts\python scripts\s3_report.py
.venv\Scripts\python scripts\play.py --who teacher --seed 5001 --record artifacts/s3_teacher.mp4 --no-window
.venv\Scripts\python scripts\play.py --who runs/bc0.pt --seed 5001 --record artifacts/s3_student.mp4 --no-window
.venv\Scripts\python scripts\play.py --who runs/bc0.pt --seed 5002
```

Последняя команда открывает окно с живой игрой ученика.

## 7. Продолжение работы с Claude Code

Откройте папку проекта в Claude Code и начните с сообщения:

> Прочитай docs/handoff-2026-10-21-windows.md и продолжай.

В handoff описано, где остановились, правила работы по срезам и что делать дальше.

## Видео

Видео пишутся в H.264 через ffmpeg (NVENC, если есть видеокарта NVIDIA), иначе запасным кодеком OpenCV, который не открывается на iPhone. Поставьте ffmpeg и добавьте его в PATH.
