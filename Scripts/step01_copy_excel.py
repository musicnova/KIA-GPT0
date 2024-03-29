# -*- coding: utf-8 -*-

class step01_copy_excel():
    def __init__(self):
        pass

    def run(self, msg):
        # 1. Автомобили Kia в кредит - расчитать кредит на покупку https://www.kia.ru/buy/calc/
        # 2. Расчет стоимости ТО - https://www.kia.ru/service/calculator_to/
        # 3. Программа Trade-In - https://www.kia.ru/buy/trade-in/
        EXCEL_TEXT = ""
        import configparser
        import pathlib
        import os
        config = configparser.ConfigParser()
        config.read('config.txt')
        knowledge_dir = config["COLAB"]["knowledge_dir"]
        if knowledge_dir is None: knowledge_dir = "./knowledge"
        pathlib.Path(knowledge_dir).mkdir(parents=True, exist_ok=True)
        filepath_00 = os.path.join(knowledge_dir, "COPY.md")
        with open(filepath_00, "w") as f00:
            f00.write("# section-0\n")
        filepath_01 = os.path.join(knowledge_dir, "COPY.md")
        with open(filepath_01, "a") as f01:
            f01.write("## 01\n")
            f01.write("""Таблица: "?", Цена: "?"\n""")
            f01.write("###\n")
        filepath_02 = os.path.join(knowledge_dir, "COPY.md")
        with open(filepath_02, "a") as f02:
            f02.write("## 02\n")
            f02.write("""Таблица: "?", Цена: "?"\n""")
            f02.write("###\n")
        filepath_03 = os.path.join(knowledge_dir, "COPY.md")
        with open(filepath_03, "a") as f03:
            f03.write("## 03\n")
            f03.write("""Таблица: "?", Цена: "?"\n""")
            f03.write("###\n")

        print(msg, " ... OK")

if __name__ == '__main__':
    step01_copy_excel().run("")
