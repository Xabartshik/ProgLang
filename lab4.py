import sys
import datetime

from lib.CFGParser import CFGParser


class FileLogger:
    """Класс для одновременного вывода в консоль и файл"""

    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w', encoding='utf-8')
        self.original_stdout = sys.stdout

        # Записываем заголовок файла
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.file.write("=" * 80 + "\n")
        self.file.write("ГЕНЕРАЦИЯ ДЕРЕВЬЕВ ВЫВОДА\n")
        self.file.write("=" * 80 + "\n")
        self.file.write(f"Дата выполнения: {timestamp}\n\n")

    def write(self, text):
        # Выводим в консоль
        self.original_stdout.write(text)
        # Записываем в файл
        self.file.write(text)

    def flush(self):
        self.original_stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.original_stdout



# Запускаем логирование
logger = FileLogger("lab4_results.txt")
sys.stdout = logger

try:
    # Пример использования для данной грамматики
    rules = {
        'S': ['aSbS', 'bSaS', '']
    }
    terminals = {'a', 'b'}
    nonterminals = {'S'}

    parser = CFGParser(rules, nonterminals, terminals)

    print("ГЕНЕРАЦИЯ ДЕРЕВЬЕВ ВЫВОДА")
    print("=" * 60)
    target_string = "abab"
    parser.print_parse_trees(target_string)

finally:
    # Закрываем логгер и восстанавливаем stdout
    logger.close()