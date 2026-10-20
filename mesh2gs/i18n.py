"""Interface language: the code is in English, the Russian translation lives here.

Blender looks every interface string up in bpy.app.translations for its current locale.
The add-on's preference picks where the Russian table goes: under ru_RU only (AUTO,
follow Blender), under every locale (RU, Russian whatever Blender's language), or
nowhere (EN). Strings built at run time go through tr(); they carry "{}" fields and are
formatted after the lookup.
"""

import bpy

# Every add-on string (properties, operators, panels, labels) carries this translation context,
# so the table never touches Blender's own strings: "Scale" in the Transform panel stays as is.
CTX = "Mesh2GS"

RU = {
    # methods and settings
    "Method": "Метод",
    "M1 — analytic": "M1 — аналитический",
    "Points spread by surface area, color sampled at every point; no GPU":
        "Точки по площади поверхности, цвет в каждой точке; без GPU",
    "M0 — KIRI scheme": "M0 — схема KIRI",
    "Fixed Gaussians per triangle, like Mesh 2 3DGS in KIRI 3DGS Render":
        "Фиксированные гауссианы на треугольник, как Mesh 2 3DGS в KIRI 3DGS Render",
    "M2 — optimization": "M2 — оптимизация",
    "3DGS training from scratch on rendered views; needs NVIDIA (CUDA) and a Python with gsplat":
        "Обучение 3DGS с нуля на рендерах ракурсов; нужна NVIDIA (CUDA) и Python с gsplat",
    "M3 — hybrid": "M3 — гибрид",
    "M1 as the start + a short training on rendered views; needs NVIDIA (CUDA) and a Python with gsplat":
        "M1 как старт + короткое обучение на рендерах; нужна NVIDIA (CUDA) и Python с gsplat",
    "Budget": "Бюджет",
    "Count": "Число",
    "Set the total number of Gaussians": "Задать итоговое число гауссиан",
    "Density": "Плотность",
    "Set the number of Gaussians per square meter of surface": "Задать число гауссиан на квадратный метр поверхности",
    "Gaussians": "Гауссиан",
    "How many Gaussians to build over all selected objects": "Сколько гауссиан построить на всех выделенных объектах",
    "Per m²": "На м²",
    "Gaussians per square meter of surface": "Число гауссиан на квадратный метр поверхности",
    "Per triangle": "На треугольник",
    "How many Gaussians to put in every triangle (KIRI actually puts 1)":
        "Сколько гауссиан ставить в каждый треугольник (KIRI фактически ставит 1)",
    "Like KIRI": "Как KIRI",
    "Color": "Цвет",
    "Albedo": "Альбедо",
    "Material color only, no light: the object can be relit": "Только цвет материала, без света: объект можно переосвещать",
    "Scene light": "Свет сцены",
    "Bake the scene's lighting (Cycles) into the colors: the object looks as rendered; needed to fit it "
    "into a captured splat scene":
        "Запечь освещение сцены (Cycles) в цвета: объект выглядит как на рендере; нужно, чтобы вписать его "
        "в снятую сплат-сцену",
    "Splat scene": "Сплат-сцена",
    "A captured scene (imported PLY/SPZ) whose light to bake into the object":
        "Снятая сцена (импортированный PLY/SPZ), чей свет запекать в объект",
    "Light samples": "Сэмплы света",
    "Cycles samples for the lighting bake: more is less noise and slower":
        "Сэмплов Cycles при запекании освещения: больше — меньше шума, дольше",
    "Bake procedural": "Запекать процедурные",
    "A procedural Base Color is baked into a texture by Cycles on a temporary copy of the object":
        "Процедурный Base Color запекается в текстуру через Cycles на временной копии объекта",
    "Include glow": "Учитывать свечение",
    "Add the material's Emission to the color: screens, indicators, lamps. Needs baking":
        "Добавить к цвету свечение материала (Emission): экраны, индикаторы, лампы. Требует запекания",
    "Resolution": "Разрешение",
    "Bake texture size per object": "Размер текстуры запекания на объект",
    "Selected faces only": "Только выделенные грани",
    "In Edit Mode, convert only the selected faces": "В режиме редактирования конвертировать только выделенные грани",
    "Overlap k": "Перекрытие k",
    "Disc radius relative to the mean spacing between Gaussians": "Радиус диска относительно среднего расстояния между гауссианами",
    "Sampling": "Сэмплинг",
    "Stratified": "Стратифицированный",
    "Even spacing over the surface, no clumps or holes": "Равномерный шаг по поверхности, без сгустков и дыр",
    "Random": "Случайный",
    "Independent random points; for comparison in experiments": "Независимые случайные точки; для сравнения в экспериментах",
    "Stay inside edges": "Не выходить за рёбра",
    "Squash discs near sharp and boundary edges so they do not overhang":
        "Сплющивать диски у острых и граничных рёбер, чтобы они не свешивались за край",
    "Sharp edge from": "Острое ребро от",
    "Angle between faces from which an edge counts as sharp": "Угол между гранями, начиная с которого ребро считается острым",
    "Random generator seed, for reproducible results": "Зерно генератора случайных чисел, для воспроизводимых результатов",
    "Show up to": "Показывать до",
    "How many Gaussians to draw in the viewport; export is not affected":
        "Сколько гауссиан рисовать во вьюпорте; на экспорт не влияет",
    "Iterations": "Итераций",
    "M2 training steps from a random start; 7000 as in the experiments":
        "Шагов обучения M2 со случайного старта; 7000 — как в экспериментах",
    "M3 refinement steps after M1; 1000 beat M1 on every benchmark model":
        "Шагов дообучения M3 после M1; 1000 хватило, чтобы обойти M1 на всех моделях бенчмарка",
    "Views": "Ракурсов",
    "How many renders from cameras on a sphere around the objects the training sees":
        "Сколько рендеров с камер на сфере вокруг объектов видит обучение",
    "Render size": "Размер рендера",
    "Side of the square view render, in pixels": "Сторона квадратного рендера ракурса в пикселях",
    "Hide source objects": "Скрыть исходные объекты",
    "Hide the source objects after converting, to see the result":
        "После конвертации скрыть исходные объекты, чтобы видеть результат",
    # preferences
    "Add-on language": "Язык аддона",
    "Language of the Mesh2GS interface (Blender's own interface is not affected). "
    "Blender 3.6 needs Interface → Translation → Interface turned on for it":
        "Язык интерфейса Mesh2GS (на интерфейс самого Blender не влияет). "
        "В Blender 3.6 для этого нужно включить Интерфейс → Перевод → Интерфейс",
    "As Blender": "Как в Blender",
    "Russian if Blender's interface is Russian, English otherwise":
        "Русский, если интерфейс Blender русский, иначе английский",
    "Always Russian": "Всегда русский",
    "Always English": "Всегда английский",
    "Python with gsplat": "Python с gsplat",
    "Interpreter with PyTorch (CUDA) and gsplat installed; M2/M3 train in it. "
    "Empty: the development environment in the repository, if there is one":
        "Интерпретатор, в котором установлены PyTorch (CUDA) и gsplat; в нём идёт обучение M2/M3. "
        "Пусто — окружение разработки в репозитории, если оно есть",
    "In use: {}": "Используется: {}",
    "No Python set: M2 and M3 are unavailable": "Python не задан: M2 и M3 недоступны",
    "M0 and M1 work without it, on any OS and graphics card.": "M0 и M1 работают без него, на любой ОС и видеокарте.",
    "How to prepare the Python for M2, M3 and light from a splat scene:":
        "Как подготовить Python для M2, M3 и света из сплат-сцены:",
    "Needs an NVIDIA graphics card, Python {} and ≈5 GB of disk": "Нужны видеокарта NVIDIA, Python {} и ≈5 ГБ на диске",
    "1. Install Python 3.10 and the NVIDIA driver.": "1. Установите Python 3.10 и драйвер NVIDIA.",
    "2. Press Copy commands and run them in a terminal in any folder:":
        "2. Нажмите «Скопировать команды» и выполните их в терминале в любой папке:",
    "    this makes mesh2gs-venv with PyTorch (CUDA 12.4) and gsplat (≈5 GB, a few minutes).":
        "    создастся mesh2gs-venv с PyTorch (CUDA 12.4) и gsplat (≈5 ГБ, несколько минут).",
    "3. Set the field above to mesh2gs-venv/Scripts/python.exe (Linux: mesh2gs-venv/bin/python).":
        "3. В поле выше укажите mesh2gs-venv/Scripts/python.exe (Linux: mesh2gs-venv/bin/python).",
    "4. Press Check: the graphics card's model should appear.": "4. Нажмите «Проверить»: должна появиться модель видеокарты.",
    "Python file not found": "файл Python не найден",
    "exit code {}": "код выхода {}",
    "CUDA unavailable": "CUDA недоступна",
    # panels
    "Objects: {}": "Объектов: {}",
    "Triangles: {}": "Треугольников: {}",
    "Area: {:.2f} m²": "Площадь: {:.2f} м²",
    "All materials: scene light → bake {}²": "Все материалы: свет сцены → bake {}²",
    "… {} more materials": "… ещё материалов: {}",
    "no material": "без материала",
    "{}: color + glow → bake {}²": "{}: цвет + свечение → bake {}²",
    "{}: color": "{}: цвет",
    "{}: {} → bake {}²": "{}: {} → bake {}²",
    "{}: {}, viewport color": "{}: {}, цвет вьюпорта",
    "Select mesh objects": "Выделите mesh-объекты",
    "This method comes in a later version": "Метод будет в следующей версии",
    "Light from a splat scene (optional):": "Свет из сплат-сцены (необязательно):",
    "≈ {} Gaussians · {:.1f} MB PLY": "≈ {} гауссиан · {:.1f} МБ PLY",
    "Training: NVIDIA GPU, {} views": "Обучение: GPU NVIDIA, ракурсов {}",
    "Needs a Python with PyTorch and gsplat:": "Нужен Python с PyTorch и gsplat:",
    "Edit → Preferences → Add-ons → Mesh2GS": "Правка → Настройки → Дополнения → Mesh2GS",
    "1/2 rendering views (Esc cancels)": "1/2 рендер ракурсов (Esc — отмена)",
    "2/2 training: you can keep working": "2/2 обучение — можно работать дальше",
    "{:.0f} s so far": "Прошло {:.0f} с",
    "Convert": "Конвертировать",
    "Advanced": "Дополнительно",
    "M3 starts from M1 with these settings:": "Старт M3 — M1 с настройками ниже:",
    "Result": "Результат",
    "Method: {}": "Метод: {}",
    "Gaussians: {}": "Гауссиан: {}",
    "Time: {:.2f} s": "Время: {:.2f} с",
    "PLY: {:.1f} MB": "PLY: {:.1f} МБ",
    "Export PLY…": "Экспорт PLY…",
    "Export SPZ…": "Экспорт SPZ…",
    "Mesh2GS: to Gaussians…": "Mesh2GS: в гауссианы…",
    # operators
    "Convert to Gaussians": "Конвертировать в гауссианы",
    "Build Gaussians on the selected mesh objects with the chosen method":
        "Построить гауссианы на выделенных mesh-объектах выбранным методом",
    "Select at least one mesh object": "Выделите хотя бы один mesh-объект",
    "Training is running; wait for it or cancel it": "Идёт обучение; дождитесь конца или отмените",
    "Set a Python with PyTorch and gsplat in the add-on preferences": "Укажите Python с PyTorch и gsplat в настройках аддона",
    "{}: {} Gaussians in {:.2f} s": "{}: {} гауссиан за {:.2f} с",
    "{}: {} Gaussians in {:.1f} s": "{}: {} гауссиан за {:.1f} с",
    "M2 and M3 work in Object Mode": "M2 и M3 работают в режиме объекта",
    "Esc cancels": "Esc — отмена",
    "cancel on the Mesh2GS panel": "отмена — на панели Mesh2GS",
    "{}: cancelled": "{}: отменено",
    "Light from splat scene": "Свет из сплат-сцены",
    "Render a 360° panorama of the chosen splat scene from the center of the selected objects and make it the "
    "World; the Scene light color mode then bakes the scene's own light into the object. Needs a Python with "
    "PyTorch and gsplat (NVIDIA)":
        "Снять панораму 360° выбранной сплат-сцены из центра выделенных объектов и поставить её в World; режим "
        "«Свет сцены» запечёт в объект свет самой сцены. Нужен Python с PyTorch и gsplat (NVIDIA)",
    "Choose a splat scene (an imported PLY/SPZ or a conversion result)":
        "Выберите сплат-сцену (импорт PLY/SPZ или результат конвертации)",
    "Select the objects that go into the scene": "Выделите объекты, которые встанут в сцену",
    "World: panorama of “{}” from ({:.2f}, {:.2f}, {:.2f}) in {:.1f} s; color mode set to Scene light":
        "World: панорама «{}» из точки ({:.2f}, {:.2f}, {:.2f}) за {:.1f} с; режим цвета — «Свет сцены»",
    "Cancel training": "Отменить обучение",
    "Stop the M2/M3 training; no result will appear": "Остановить обучение M2/M3; результат не появится",
    "Copy commands": "Скопировать команды",
    "Copy to the clipboard the commands that make a Python with PyTorch and gsplat (mesh2gs-venv)":
        "Скопировать в буфер обмена команды, которые создают Python с PyTorch и gsplat (mesh2gs-venv)",
    "{} commands copied: paste them into a terminal": "Скопировано команд: {} — вставьте их в терминал",
    "Check": "Проверить",
    "Run the set Python and check PyTorch, gsplat and the CUDA graphics card (up to a minute)":
        "Запустить заданный Python и проверить PyTorch, gsplat и видеокарту CUDA (до минуты)",
    "Source ↔ Gaussians": "Исходник ↔ гауссианы",
    "Switch visibility between the source objects and the result": "Переключить видимость между исходными объектами и результатом",
    "Export PLY": "Экспорт PLY",
    "Save the active object's Gaussians as a 3DGS PLY": "Сохранить гауссианы активного объекта в PLY формата 3DGS",
    "Axes": "Оси",
    "The file's coordinate system: where +X, +Y, +Z point": "Система координат файла: куда смотрят +X, +Y, +Z",
    "Blender, no rotation": "Blender, без поворота",
    "RFU — Blender, no rotation": "RFU — Blender, без поворота",
    "Saved {} Gaussians, {:.1f} MB": "Сохранено {} гауссиан, {:.1f} МБ",
    "Export SPZ": "Экспорт SPZ",
    "Save the active object's Gaussians as compressed SPZ (Niantic); the format fixes the axes (RUB), "
    "viewers convert them themselves":
        "Сохранить гауссианы активного объекта в сжатый формат SPZ (Niantic); оси задаёт сам формат (RUB), "
        "просмотрщики переводят их сами",
    "Scale": "Масштаб",
    "Multiply sizes when writing. SPZ stores coordinates in 0.24 mm steps within ±2048: a model in centimeters "
    "or a few centimeters across is better scaled up":
        "Умножить размеры при записи. SPZ хранит координаты с шагом 0,24 мм в пределах ±2048: модель "
        "в сантиметрах или в несколько сантиметров лучше увеличить",
    "Coordinates reach {:.0f}: SPZ only holds ±{:.0f}, scale down":
        "Координаты до {:.0f}: SPZ хранит только ±{:.0f}, уменьшите масштаб",
    "The model is {:.1f} cm: SPZ's 0.24 mm step will smooth fine detail; scale it up or save PLY":
        "Модель {:.1f} см: шаг SPZ 0,24 мм, мелкие детали сгладятся — увеличьте масштаб или сохраните PLY",
    "Import PLY (3DGS)": "Импорт PLY (3DGS)",
    "Load splats from a 3DGS PLY (a KIRI result, say) with a viewport preview":
        "Загрузить сплаты из PLY формата 3DGS (например, результат KIRI) с превью во вьюпорте",
    "Loaded {} Gaussians": "Загружено {} гауссиан",
    "Import SPZ": "Импорт SPZ",
    "Load splats from compressed SPZ (versions 2 and 3) with a viewport preview":
        "Загрузить сплаты из сжатого SPZ (версии 2 и 3) с превью во вьюпорте",
    # installing the components
    "Install to": "Установить в",
    "Folder for the components of M2/M3 (≈5 GB; 9 GB free needed while installing). "
    "Empty: the add-on's own user folder":
        "Папка для компонентов M2/M3 (≈5 ГБ; во время установки нужно 9 ГБ свободного места). "
        "Пусто — пользовательская папка аддона",
    "Install by hand": "Установить вручную",
    "Show the commands to make the training Python yourself": "Показать команды, чтобы подготовить Python самому",
    "Components for M2, M3 and light from a splat scene": "Компоненты для M2, M3 и света из сплат-сцены",
    "PyTorch and gsplat train Gaussians on an NVIDIA graphics card.":
        "PyTorch и gsplat обучают гауссианы на видеокарте NVIDIA.",
    "M0 and M1 work without them, on any computer.": "M0 и M1 работают без них, на любом компьютере.",
    "Needs Windows or Linux with an NVIDIA graphics card": "Нужны Windows или Linux и видеокарта NVIDIA",
    "Into {} · {:.1f} GB free": "В {} · свободно {:.1f} ГБ",
    "Installed: {}": "Установлено: {}",
    "Install components (≈5 GB)": "Установить компоненты (≈5 ГБ)",
    "Download a Python 3.10 of its own, PyTorch with CUDA and gsplat into the chosen folder and use them for "
    "M2, M3 and the splat-scene light. Takes several minutes; Blender stays usable":
        "Скачать в выбранную папку собственный Python 3.10, PyTorch с CUDA и gsplat и использовать их для "
        "M2, M3 и света из сплат-сцены. Займёт несколько минут; Blender остаётся рабочим",
    "Installation is running": "Идёт установка",
    "Installing into {}": "Установка в {}",
    "Cancel installation": "Отменить установку",
    "Stop downloading and installing the components; what is already there stays":
        "Остановить загрузку и установку компонентов; уже установленное останется",
    "downloading Python 3.10": "загрузка Python 3.10",
    "unpacking Python": "распаковка Python",
    "installing PyTorch (≈2.5 GB)": "установка PyTorch (≈2,5 ГБ)",
    "installing gsplat": "установка gsplat",
    "checking the graphics card": "проверка видеокарты",
    "{}/{}: {}": "{}/{}: {}",
    "Done": "Готово",
    "Cancelled": "Отменено",
    "Failed": "Ошибка",
    "only Windows and Linux on x86-64 with an NVIDIA graphics card":
        "только Windows и Linux на x86-64 с видеокартой NVIDIA",
    "{:.1f} GB free on that drive, {:.0f} GB needed": "на этом диске свободно {:.1f} ГБ, нужно {:.0f} ГБ",
    "the downloaded Python does not match its checksum": "скачанный Python не совпал с контрольной суммой",
    "Needs PyTorch and gsplat (one button, ≈5 GB):": "Нужны PyTorch и gsplat (одна кнопка, ≈5 ГБ):",
    "Install components…": "Установить компоненты…",
    # materials, extraction, training
    "no Principled BSDF": "нет Principled BSDF",
    "procedural color": "процедурный цвет",
    "texture through Mapping": "текстура через Mapping",
    "image “{}” is not loaded": "изображение «{}» не загружено",
    "nothing to convert: no triangles": "нет треугольников для конвертации",
    "{}: the mesh has no UV map for “{}”, the viewport color is used":
        "{}: у меша нет UV-карты для «{}», взят цвет вьюпорта",
    "{}: {}, the viewport color is used": "{}: {} — взят цвет вьюпорта",
    "no Python with PyTorch and gsplat is set (Mesh2GS add-on preferences)":
        "не задан Python с PyTorch и gsplat (настройки аддона Mesh2GS)",
    "panorama: exit code {}": "панорама: код выхода {}",
    "{}: views {}/{}": "{}: ракурсы {}/{}",
    "{}: starting PyTorch…": "{}: запуск PyTorch…",
    "{}: training {}/{}": "{}: обучение {}/{}",
    "the trainer exited with code {}": "тренер завершился с кодом {}",
    "needs an NVIDIA graphics card with CUDA: torch.cuda.is_available() = False":
        "нужна видеокарта NVIDIA с CUDA: torch.cuda.is_available() = False",
    "mesh has no triangles with non-zero area": "у меша нет треугольников с ненулевой площадью",
}


def tr(text):
    """`text` in the interface language (looked up as is; format after)."""
    return bpy.app.translations.pgettext_iface(text, CTX)


def interface_strings(cls):
    """(names, tooltips) of an add-on class: its label, and its own properties and enum items."""
    names, tips = set(), set()
    if getattr(cls, "bl_label", ""):
        names.add(cls.bl_label)
    if getattr(cls, "bl_description", ""):
        tips.add(cls.bl_description)
    rna = cls.bl_rna
    if issubclass(cls, bpy.types.Operator):  # operator properties live on the operator's type
        module, name = cls.bl_idname.split(".")
        rna = getattr(getattr(bpy.ops, module), name).get_rna_type()
    for identifier in getattr(cls, "__annotations__", {}):
        prop = rna.properties[identifier]
        if prop.is_hidden:
            continue
        names.add(prop.name)
        tips.add(prop.description)
        for item in getattr(prop, "enum_items", []):
            names.add(item.name)
            tips.add(item.description)
    return names, tips - {""}


def _tooltips():
    """Every tooltip text of the add-on's registered classes."""
    import sys
    tips = set()
    for cls in getattr(sys.modules[__package__], "classes", ()):
        tips |= interface_strings(cls)[1]
    return tips


def _table():
    # Tooltips are looked up in Blender's default context whatever the property's context
    # is; they are whole sentences of the add-on's own, so they cannot catch Blender strings.
    table = {(CTX, en): ru for en, ru in RU.items()}
    table.update({("*", en): RU[en] for en in _tooltips() if en in RU})
    return table


def apply(language='AUTO'):
    """Register the Russian table for `language` ('AUTO', 'RU' or 'EN')."""
    unregister()
    if language == 'EN':
        return
    locales = ["ru_RU"] if language == 'AUTO' else sorted(set(bpy.app.translations.locales) | {"en_US"})
    table = _table()
    bpy.app.translations.register(__package__, {locale: table for locale in locales})


def unregister():
    try:
        bpy.app.translations.unregister(__package__)
    except (ValueError, RuntimeError):
        pass


def preference(context=None):
    """The language preference; 'AUTO' while the add-on has no preferences yet (or none at
    all, when registered by a script rather than enabled)."""
    addon = (context or bpy.context).preferences.addons.get(__package__)
    prefs = getattr(addon, "preferences", None)
    return prefs.language if prefs is not None else 'AUTO'
