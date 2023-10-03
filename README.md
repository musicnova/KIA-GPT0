# KIA-GPT

**Тестирование:** 
https://docs.google.com/spreadsheets/d/1dlzmaiKBX8ZxIc-OM9XUrv2vkQVgY1Gs5SYPm9ZZQ8s/edit#gid=1214543959

**База знаний** 
knowledge/database.txt

**Коллаб для использования** 
https://colab.research.google.com/drive/1sY-PdWx6TnzMvQDI5wxegFt8USWlDiN3?usp=sharing


Редактор онлайн: https://6d34f4ce7edb.vps.myjino.ru/


**Памятка по работе в GIT**

Рекомендации для разработчиков:
1. Ищите задачу (Issue), назначаете себя исполнителем (Assignee).
2. Делаете ветку (Branch). Скачиваете локально с помощью GitHub Desktop или другими программами, переходите в свою ветку.
3. Делаете копию knowledge/main.ipynb в knowledge/<имя ветки>.ipynb и сверяетесь с последней версией коллаба от группы алгоритма.
4. Вносите правки локально. Сохраняете изменения (Commit) с помощью GitHub Desktop или другими программами. Экспортируете в GitHub (Push).
5. Готовите показ группе тестирования (Pull Request). Сообщаете координатору группы тестирования (Галине).

Рекомендации для тестировщиков:
1. Используйте только тот Excel c вопросами, что указывает координатор группы тестирования (Галина).
2. Основную ветку main тестируйте через телеграмм бота. При обнаружении багов создавайте задачи (Issue).
3. Ветки с правками (bugfix) и сборки (release) тестируете коллабом. Если его нет в ветке, то пользуйтесь последней версией коллаба от группы алгоритма.
4. После завершения тестирования ветки сообщайте координатору группы тестирования (Галине). Только она принимает решение об изменениях MD файла. И ещё Светлана.
5. Если MD файл не меняется, то можно попросить подтвердить (Merge) изменения координатора вашей группы.

Рекомендации для руководителей:
1. Проверьте, что все изменения залиты в GIT в knowledge.
2. Помогите участникам группы с установкой программ.
3. Выполните за них заливку файлов в GIT если потребуется.
4. С помощью редакторов VSCODE или PyCharm удалите противоречия (Conflict) из файла, если его редактировали разные участники.
5. Сообщите группе тестирования о закрытии задач (Issue) через комментарии.

Рекомендации для тех, кто не как все:
1. Выше есть полезные ссылки, где что лежит.
2. Можно создать свой MD файл в своей ветке release/...
3. После завершения тестирования сообщить об этом координатору группы тестирования (Галине).
4. Если вы подняли свой телеграмм бот, поделитесь со всеми.
5. Если вы узнали что-то интересное, опишите это в задаче (Issue) или в гипотезах (Discussions).

## Сплитеры группы парсер

`Сейчас используется стандартный Markdown Splitter. В частности h1, h2, h3 и ссылки с всплывающими подсказками в [](). И иногда используются сплиттеры УИИ _# , ##_.`

`В версии V1 парсится сайт под стандартный Markdown Splitter. Скачиваются PDF файлы. Далее нейрокопирайтером обрабатываются тексты PDF файлов. Cплошные тексты на более чем 1000 токенов тоже перерабатываются нейрокопирайтером. В PDF файлах сплиттеры УИИ заменяются на стандартные h1, h2, h3. В остальных файлах разметка остается смешанной: сплиттеры УИИ _# , ##_ и h1, h2, h3.`

## Сплитеры группы диалоги

`Сейчас используется стандартный Markdown Splitter. И иногда используются сплиттеры УИИ _# , ##_.`

`Выделяются диалоги по разным тематикам. Далее нейрокопирайтером обрабатываются тексты диалогов. Удаляются ненужные фрагменты и сплиттеры УИИ заменяются на стандартные h1, h2, h3. Но пока не везде и не всегда, где-то тематика сохраняется в h1, а разметка УИИ _# , ##_ остается. В итоге в файлах разметка остается смешанной: сплиттеры УИИ _# , ##_ и h1, h2, h3.`

## Сплитеры группы видео

`Сейчас используется стандартный Markdown Splitter. В частности h1, h2, h3 и ссылки с всплывающими подсказками в []().`

`Выделяются видео транскрибации по разным тематикам. Далее нейрокопирайтером обрабатываются тексты видео транскрибаций. Удаляются ненужные фрагменты и сплиттеры УИИ _# , ##_ заменяются на стандартные h1, h2, h3.`

## Сплитеры группы алгоритма

`Сейчас используется стандартный Markdown Splitter. И дополнительно <chunk>.`

`Базы знаний объединяются в одну. При объединении рекомендуется добавлять заголовок h1 между частями, например, # 1, # 2, # 3, # 4, так в некоторых группах разметки УИИ идут раньше h1, h2, h3.`

`Далее общий текст делится по заголовкам h1, h2, h3 и сплиттерам УИИ _# , ##_ c помощью разделителя <chunk>. Далее ещё раз делится по 1000 токенов с помощью разделителя <chunk>. В начале добавляется заголовок из предыдущего h1 (не путать со сплиттером УИИ _# , ##_), в конце добавляются пробелы, если overlap не ноль.`
