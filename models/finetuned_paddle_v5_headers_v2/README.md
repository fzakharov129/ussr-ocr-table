# PP-OCRv5 headers v2

Распознаватель готовых строк и однострочных ячеек. Весами для запуска служит `best_model.pdparams`; `last_model.pdparams`, `history.csv` и `final_metrics.json` сохранены как результаты обучения.

В среде CPU из корня проекта:

```powershell
python -m pip install -r .\models\finetuned_paddle_v5_headers_v2\requirements_cpu.txt
python .\models\finetuned_paddle_v5_headers_v2\recognize.py "D:\path\to\line.png"
```

Вывод — JSON с распознанным текстом и уверенностью. Для последних весов добавьте `--weights last_model.pdparams`. Данные для запуска: `inference_config.yaml`, `ppocrv5_cyrillic_dict.txt` и локальные исходники `PaddleOCR-3.3.0-runtime.zip`. При первом запуске исходники извлекаются в `.paddleocr_source`.

Конфигурация и неизменённый словарь взяты из headers v1, от которого дообучалась эта модель. Это не оригинальный `training_config.yaml` сервера; по возможности сохраните его для сверки. Для полного возобновления обучения также потребуются серверный workspace и latest checkpoint с оптимизатором. Качество распознавания и список ошибок в этом комплекте не проверялись.

Основной pipeline выбирает headers v2 по умолчанию; для запуска также нужен комплект values v2. Для многострочных ячеек необходим отдельный детектор строк.
