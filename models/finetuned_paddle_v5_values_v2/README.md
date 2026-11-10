# PP-OCRv5 values v2

Распознаватель готовых изображений числовых ячеек. `best_model.pdparams` используется по умолчанию; `last_model.pdparams`, `history.csv` и `final_metrics.json` сохранены как результаты обучения.

Из корня проекта в активной CPU-среде:

```powershell
python -m pip install -r .\models\finetuned_paddle_v5_values_v2\requirements_cpu.txt
python .\models\finetuned_paddle_v5_values_v2\recognize.py "D:\path\to\value.png"
```

Скрипт возвращает JSON с распознанным текстом и уверенностью. Для последних весов добавьте `--weights last_model.pdparams`. Конфигурация разрешает цифры и точку; число точек не ограничено. Для запуска нужны `inference_config.yaml`, `ppocrv5_cyrillic_dict.txt` и `PaddleOCR-3.3.0-runtime.zip`. При первом запуске исходники извлекаются в `.paddleocr_source`.

Конфигурация и полный словарь без изменения порядка символов взяты из values v1, от которой дообучалась модель. Это не оригинальный `training_config.yaml` сервера; сохраните его для сверки, если он доступен. Для полного продолжения обучения нужны серверный workspace и latest checkpoint с оптимизатором. Совместимость имён и размеров параметров проверяется при загрузке.

Основной pipeline `src/ocr.py` выбирает values v2 по умолчанию. Распознавание изображений и список ошибок при подготовке комплекта не проверялись.
