"""Запуск обработки книги из ZIP с контрольными точками."""

import argparse
import sys

from src import PipelineStage, run_pipeline, USSRTableError


def main() -> None:
    # Windows redirected output may otherwise use an encoding without Cyrillic.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and not stream.isatty():
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Распознавание таблиц из ZIP с изображениями страниц.")
    parser.add_argument("zip_path", help="Путь к ZIP с книгой")
    parser.add_argument("--start-stage", choices=[s.value for s in PipelineStage],
                        default=PipelineStage.DETECTION.value,
                        help="Этап начала; по умолчанию detection")
    parser.add_argument("--end-stage", choices=[s.value for s in PipelineStage],
                        default=PipelineStage.EXPORT.value,
                        help="Последний этап включительно; по умолчанию export")
    parser.add_argument("--outputs-dir", help="Родительская папка результатов; по умолчанию outputs в проекте")
    parser.add_argument("--book-id", help="Название папки книги; по умолчанию имя ZIP без расширения")
    parser.add_argument("--no-progress", action="store_true", help="Отключить полосы прогресса")
    parser.add_argument("--log-requests", action="store_true",
                        help="Выводить запросы и ответы VLM в терминал")
    args = parser.parse_args()
    try:
        paths = run_pipeline(args.zip_path, start_stage=args.start_stage,
                             end_stage=args.end_stage,
                             outputs_dir=args.outputs_dir, book_id=args.book_id,
                             show_progress=not args.no_progress,
                             log_requests=args.log_requests)
    except (USSRTableError, OSError, ValueError) as error:
        parser.exit(1, f"Ошибка: {error}\n")
    if "checkpoint" in paths:
        print(f"Снимок этапа: {paths['checkpoint']}")
    else:
        print(f"JSON: {paths['json']}")
        print(f"XLSX: {paths['xlsx']}")
        print(f"Таблицы страниц: {paths['pages_dir']}")


if __name__ == "__main__":
    main()
