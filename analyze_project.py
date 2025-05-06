import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set


def analyze_python_file(file_path: str) -> List[str]:
    """Анализирует Python файл и извлекает информацию о функциях и классах."""
    results = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        try:
            tree = ast.parse(content)

            # Словарь для хранения классов и их методов
            class_methods = {}

            for node in ast.walk(tree):
                # Обработка функций верхнего уровня
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # Пропускаем методы классов (они будут обработаны отдельно)
                    if isinstance(node.parent, ast.ClassDef) if hasattr(node, 'parent') else False:
                        continue

                    args_info = []
                    returns_info = None

                    # Получаем аргументы
                    for arg in node.args.args:
                        if hasattr(arg, 'annotation') and arg.annotation:
                            if isinstance(arg.annotation, ast.Name):
                                args_info.append(f"{arg.arg}: {arg.annotation.id}")
                            elif isinstance(arg.annotation, ast.Attribute):
                                args_info.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
                            else:
                                args_info.append(arg.arg)
                        else:
                            args_info.append(arg.arg)

                    # Получаем возвращаемое значение
                    if node.returns:
                        if isinstance(node.returns, ast.Name):
                            returns_info = node.returns.id
                        elif isinstance(node.returns, ast.Attribute):
                            returns_info = ast.unparse(node.returns)
                        else:
                            returns_info = ast.unparse(node.returns)

                    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
                    func_signature = f"{prefix}def {node.name}({', '.join(args_info)})"
                    if returns_info:
                        func_signature += f" -> {returns_info}"

                    results.append(func_signature)

                # Обработка классов
                elif isinstance(node, ast.ClassDef):
                    class_info = f"class {node.name}"
                    if node.bases:
                        bases = []
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                bases.append(base.id)
                            elif isinstance(base, ast.Attribute):
                                bases.append(ast.unparse(base))
                            else:
                                bases.append(ast.unparse(base))
                        class_info += f"({', '.join(bases)})"

                    results.append(class_info)
                    class_methods[node.name] = []

                    # Ищем методы класса
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                            args_info = []
                            returns_info = None

                            # Получаем аргументы
                            for arg in item.args.args:
                                if arg.arg == 'self' or arg.arg == 'cls':
                                    args_info.append(arg.arg)
                                    continue

                                if hasattr(arg, 'annotation') and arg.annotation:
                                    if isinstance(arg.annotation, ast.Name):
                                        args_info.append(f"{arg.arg}: {arg.annotation.id}")
                                    elif isinstance(arg.annotation, ast.Attribute):
                                        args_info.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
                                    else:
                                        args_info.append(arg.arg)
                                else:
                                    args_info.append(arg.arg)

                            # Получаем возвращаемое значение
                            if item.returns:
                                if isinstance(item.returns, ast.Name):
                                    returns_info = item.returns.id
                                elif isinstance(item.returns, ast.Attribute):
                                    returns_info = ast.unparse(item.returns)
                                else:
                                    returns_info = ast.unparse(item.returns)

                            prefix = "async " if isinstance(item, ast.AsyncFunctionDef) else ""
                            method_signature = f"    {prefix}def {item.name}({', '.join(args_info)})"
                            if returns_info:
                                method_signature += f" -> {returns_info}"

                            results.append(method_signature)
                            class_methods[node.name].append(method_signature)

        except SyntaxError as e:
            # Если не удалось распарсить файл с помощью ast, используем регулярные выражения
            function_pattern = r'(async\s+)?def\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)(\s*->\s*([a-zA-Z0-9_\.]+))?'
            class_pattern = r'class\s+([a-zA-Z0-9_]+)(\(([^)]*)\))?'

            current_indent = 0
            current_class = None

            for line in content.split('\n'):
                line_stripped = line.lstrip()
                if not line_stripped or line_stripped.startswith('#'):
                    continue

                # Определяем уровень вложенности по отступам
                indent = len(line) - len(line_stripped)

                # Ищем классы
                class_match = re.search(class_pattern, line_stripped)
                if class_match:
                    current_indent = indent
                    current_class = class_match.group(1)

                    class_name = class_match.group(1)
                    bases = class_match.group(3) or ""

                    class_info = f"class {class_name}"
                    if bases:
                        class_info += f"({bases})"

                    results.append(class_info)
                    continue

                # Ищем функции
                func_match = re.search(function_pattern, line_stripped)
                if func_match:
                    async_prefix = func_match.group(1) or ""
                    func_name = func_match.group(2)
                    args = func_match.group(3) or ""
                    return_type = func_match.group(5) or ""

                    func_signature = f"{async_prefix}def {func_name}({args})"
                    if return_type:
                        func_signature += f" -> {return_type}"

                    # Если это метод класса (отступ больше, чем у класса)
                    if current_class is not None and indent > current_indent:
                        func_signature = "    " + func_signature

                    results.append(func_signature)

    except Exception as e:
        results.append(f"# Error analyzing file: {str(e)}")

    return results


def get_project_structure(root_dir: str) -> str:
    """Создает древовидную структуру проекта."""
    structure = []

    # Получаем все файлы и директории
    for root, dirs, files in os.walk(root_dir):
        level = root.replace(root_dir, '').count(os.sep)
        indent = '│   ' * (level - 1) + '├── ' if level > 0 else ''
        rel_path = os.path.basename(root)

        if level > 0:  # Пропускаем корневую директорию
            structure.append(f'{indent}{rel_path}/')

        sub_indent = '│   ' * level + '├── '

        # Сортируем файлы и директории
        files.sort()

        for i, file in enumerate(files):
            if i == len(files) - 1 and len(dirs) == 0:
                sub_indent = '│   ' * level + '└── '

            # Добавляем комментарий для JSON файлов
            if file.endswith('.json'):
                structure.append(f'{sub_indent}{file}     # Автосохранение данных')
            else:
                structure.append(f'{sub_indent}{file}')

    return '\n'.join(structure)


def analyze_project(project_dir: str) -> Tuple[Dict[str, List[str]], str]:
    """Анализирует проект и возвращает информацию о функциях и структуре."""
    file_functions = {}

    # Используем os.walk для обхода всех файлов и директорий
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, project_dir)

                # Пропускаем файлы в директории __pycache__
                if '__pycache__' in rel_path:
                    continue

                print(f"Анализ файла: {rel_path}")
                functions = analyze_python_file(file_path)
                if functions:
                    file_functions[rel_path] = functions

    structure = get_project_structure(project_dir)

    return file_functions, structure


def main():
    project_dir = input("Введите путь к директории проекта: ").strip()

    if not os.path.isdir(project_dir):
        print(f"Ошибка: {project_dir} не является директорией.")
        return

    print(f"\nАнализ проекта в {project_dir}...\n")

    file_functions, structure = analyze_project(project_dir)

    # Выводим структуру проекта
    print("Структура проекта:")
    print(structure)
    print("\n" + "=" * 50 + "\n")

    # Выводим функции по файлам
    print("Функции и классы по файлам:")
    for file_path, functions in sorted(file_functions.items()):
        print(f"\nФайл: {file_path}")
        for func in functions:
            print(f"-{func}")

    # Сохраняем результаты в файл
    output_file = os.path.join(project_dir, "project_analysis.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Структура проекта:\n")
        f.write(structure)
        f.write("\n\n" + "=" * 50 + "\n\n")
        f.write("Функции и классы по файлам:\n")
        for file_path, functions in sorted(file_functions.items()):
            f.write(f"\nФайл: {file_path}\n")
            for func in functions:
                f.write(f"-{func}\n")

    print(f"\nРезультаты анализа сохранены в {output_file}")


if __name__ == "__main__":
    main()