from flask import Flask, request, render_template_string, send_file
import json
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Свёртка матриц - сетевой проект</title>
    <style>
        body { font-family: Arial; margin: 40px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 10px; }
        textarea { width: 100%; font-family: monospace; padding: 10px; }
        input[type="number"] { width: 60px; padding: 5px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; margin: 5px; }
        .result { margin-top: 20px; padding: 15px; background: #e8f4e8; border-radius: 5px; }
        .error { background: #f8e8e8; color: red; }
        .matrix-table {
            border-collapse: collapse;
            margin: 10px 0;
            display: inline-block;
            margin-right: 30px;
        }
        .matrix-table td {
            border: 1px solid #4CAF50;
            padding: 8px 12px;
            text-align: center;
            min-width: 40px;
        }
        .flex-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }
        .params {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        .mode-selector {
            margin-bottom: 20px;
            padding: 10px;
            background: #e0e0e0;
            border-radius: 8px;
        }
        .mode-btn {
            background: #6c757d;
        }
        .mode-btn.active {
            background: #007bff;
        }
        .image-compare {
            display: flex;
            gap: 30px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .image-box {
            text-align: center;
        }
        .image-box img {
            max-width: 300px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .kernel-preset {
            margin: 10px 0;
        }
        .kernel-preset button {
            background: #28a745;
            font-size: 12px;
            padding: 5px 10px;
        }
        .layer-card {
            border: 1px solid #ccc;
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
            background: #f9f9f9;
        }
        .layer-card h4 {
            margin: 0 0 10px 0;
            color: #333;
        }
    </style>
    <script>
        function switchMode(mode) {
            document.getElementById('mode-matrix').style.display = mode === 'matrix' ? 'block' : 'none';
            document.getElementById('mode-image').style.display = mode === 'image' ? 'block' : 'none';
            document.getElementById('mode-multilayer').style.display = mode === 'multilayer' ? 'block' : 'none';
            document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelector(`.mode-btn[onclick="switchMode('${mode}')"]`).classList.add('active');
        }

        function setKernel(kernel) {
            document.getElementById('custom_kernel').value = kernel;
        }

        let layerCounter = 0;
        
        function addLayer(type) {
            layerCounter++;
            const container = document.getElementById('layers-container');
            const newLayer = document.createElement('div');
            newLayer.className = 'layer-card';
            newLayer.setAttribute('data-layer', layerCounter);
            newLayer.setAttribute('data-type', type);
            
            if (type === 'conv') {
                newLayer.innerHTML = `
                    <h4>📌 Слой ${layerCounter} (Свёртка)</h4>
                    <label>Название слоя:</label>
                    <input type="text" name="name_${layerCounter}" value="Слой ${layerCounter}" style="width: 200px;"><br><br>
                    <label>Тип: Свёртка</label><br>
                    <label>Ядро (матрица 3x3):</label><br>
                    <textarea name="kernel_${layerCounter}" rows="2" cols="40" placeholder="[[1,2,1],[2,4,2],[1,2,1]]/16"></textarea><br>
                    <button type="button" onclick="removeLayer(${layerCounter})" style="background:#dc3545; margin-top:10px;">❌ Удалить слой</button>
                `;
            } else if (type === 'pool') {
                newLayer.innerHTML = `
                    <h4>📌 Слой ${layerCounter} (MaxPooling)</h4>
                    <input type="text" name="name_${layerCounter}" value="MaxPooling ${layerCounter}" style="width: 200px;"><br><br>
                    <label>Тип: MaxPooling (уменьшает картинку в 2 раза)</label><br>
                    <input type="hidden" name="is_pool_${layerCounter}" value="true">
                    <button type="button" onclick="removeLayer(${layerCounter})" style="background:#dc3545; margin-top:10px;">❌ Удалить слой</button>
                `;
            }
            
            container.appendChild(newLayer);
        }
        
        function removeLayer(layerNum) {
            const layer = document.querySelector(`.layer-card[data-layer="${layerNum}"]`);
            if (layer) layer.remove();
        }
        
        function loadArchitecture(type) {
            document.getElementById('layers-container').innerHTML = '';
            layerCounter = 0;
        
            if (type === 'edges') {
                addLayerWithKernel('Границы', '[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]');
                addLayerWithKernel('Размытие', '[[1,1,1],[1,1,1],[1,1,1]]/9');
                addLayerWithKernel('Резкость', '[[0,-1,0],[-1,5,-1],[0,-1,0]]');
            } else if (type === 'vgg_block') {
                addLayerWithKernel('Свёртка 1', '[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]');
                addLayerWithKernel('Свёртка 2', '[[1,2,1],[2,4,2],[1,2,1]]/16');
                addPoolingLayerWithName('MaxPooling 2x2');
            } else if (type === 'blur') {
                addLayerWithKernel('Размытие 1', '[[1,1,1],[1,1,1],[1,1,1]]/9');
                addLayerWithKernel('Размытие 2', '[[1,1,1],[1,1,1],[1,1,1]]/9');
            } else if (type === 'sharpen') {
                addLayerWithKernel('Резкость', '[[0,-1,0],[-1,5,-1],[0,-1,0]]');
            }
        }
        
        function addLayerWithKernel(name, kernel) {
            layerCounter++;
            const container = document.getElementById('layers-container');
            const newLayer = document.createElement('div');
            newLayer.className = 'layer-card';
            newLayer.setAttribute('data-layer', layerCounter);
            newLayer.setAttribute('data-type', 'conv');
            newLayer.innerHTML = `
                <h4>📌 Слой ${layerCounter} (Свёртка)</h4>
                <input type="text" name="name_${layerCounter}" value="${name}" style="width: 200px;"><br><br>
                <label>Ядро (матрица 3x3):</label><br>
                <textarea name="kernel_${layerCounter}" rows="2" cols="40">${kernel}</textarea><br>
                <button type="button" onclick="removeLayer(${layerCounter})" style="background:#dc3545; margin-top:10px;">❌ Удалить</button>
            `;
            container.appendChild(newLayer);
        }
        function addPoolingLayerWithName(name) {
            layerCounter++;
            const container = document.getElementById('layers-container');
            const newLayer = document.createElement('div');
            newLayer.className = 'layer-card';
            newLayer.setAttribute('data-layer', layerCounter);
            newLayer.setAttribute('data-type', 'pool');
            newLayer.innerHTML = `
                <h4>📌 Слой ${layerCounter} (MaxPooling)</h4>
                <input type="text" name="name_${layerCounter}" value="${name}" style="width: 200px;"><br><br>
                <label>Тип: MaxPooling (уменьшает картинку в 2 раза)</label><br>
                <input type="hidden" name="is_pool_${layerCounter}" value="true">
                <button type="button" onclick="removeLayer(${layerCounter})" style="background:#dc3545; margin-top:10px;">❌ Удалить</button>
            `;
            container.appendChild(newLayer);
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>🔢 Вычисление свёртки матриц</h1>
        <p>Проект по сетевым технологиям - клиент-серверное взаимодействие</p>

        <div class="mode-selector">
            <button class="mode-btn active" onclick="switchMode('matrix')">📊 Режим: ввод матриц</button>
            <button class="mode-btn" onclick="switchMode('image')">🖼️ Режим: обработка изображений</button>
            <button class="mode-btn" onclick="switchMode('multilayer')"> Режим: конструктор CNN</button>
        </div>

        <div id="mode-matrix">
            <form method="post" action="/matrix">
                <b>Матрица A (исходная):</b><br>
                <textarea name="matrix_a" rows="4" cols="50">{{ matrix_a }}</textarea><br><br>

                <b>Ядро свёртки B:</b><br>
                <textarea name="matrix_b" rows="3" cols="30">{{ matrix_b }}</textarea><br><br>

                <div class="params">
                    <label>📐 Padding (отступы):</label>
                    <input type="number" name="padding" value="{{ padding }}" min="0" max="5">
                    <span style="margin-left: 20px;">⚡ Stride (шаг):</span>
                    <input type="number" name="stride" value="{{ stride }}" min="1" max="5">
                </div>

                <button type="submit">🚀 Вычислить свёртку</button>
            </form>

            {% if matrix_a_display %}
            <div class="flex-container">
                <div>
                    <b>Матрица A ({{ a_rows }}×{{ a_cols }}):</b><br>
                    <table class="matrix-table">
                        {% for row in matrix_a_display %}
                        <tr>
                            {% for val in row %}
                            <td>{{ val }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                <div>
                    <b>Ядро B ({{ b_rows }}×{{ b_cols }}):</b><br>
                    <table class="matrix-table">
                        {% for row in matrix_b_display %}
                        <tr>
                            {% for val in row %}
                            <td>{{ val }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                    </table>
                </div>
            </div>
            {% endif %}

            {% if result_matrix %}
            <div class="result">
                <b>Результат свёртки (padding={{ padding }}, stride={{ stride }}):</b><br>
                <table class="matrix-table">
                    {% for row in result_matrix %}
                    <tr>
                        {% for val in row %}
                        <td>{{ val }}</td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </table>
                <small>Размер: {{ res_rows }}×{{ res_cols }}</small>
            </div>
            {% endif %}
        </div>

        <div id="mode-image" style="display:none">
            <form method="post" action="/image" enctype="multipart/form-data">
                <b>Загрузите изображение:</b><br>
                <input type="file" name="image" accept="image/*" required><br><br>

                <b>Ядро свёртки (фильтр):</b><br>

                <div class="kernel-preset">
                    <b>🎯 Готовые фильтры:</b><br>
                    <button type="button" onclick="setKernel('[[0,0,0],[0,1,0],[0,0,0]]')">🔍 Без изменений</button>
                    <button type="button" onclick="setKernel('[[0,1,0],[1,-4,1],[0,1,0]]')">✏️ Выделение границ</button>
                    <button type="button" onclick="setKernel('[[1,1,1],[1,1,1],[1,1,1]]/9')">📷 Размытие</button>
                    <button type="button" onclick="setKernel('[[0,-1,0],[-1,5,-1],[0,-1,0]]')">🔪 Резкость</button>
                    <button type="button" onclick="setKernel('[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]')">🎨 Сильные границы</button>
                    <button type="button" onclick="setKernel('[[1,2,1],[2,4,2],[1,2,1]]/16')">😊 Гаусс (размытие)</button>
                </div>

                <div class="params">
                    <label>✏️ Своё ядро (через запятую или /делитель):</label><br>
                    <textarea name="custom_kernel" id="custom_kernel" rows="3" cols="40" placeholder="[[1,0,-1],[2,0,-2],[1,0,-1]]"></textarea>
                    <br><small>Пример: [[1,0,-1],[2,0,-2],[1,0,-1]]</small>
                </div>

                <button type="submit">🎨 Применить свёртку к изображению</button>
            </form>

            {% if original_image %}
            <div class="image-compare">
                <div class="image-box">
                    <b>📸 Оригинал:</b><br>
                    <img src="data:image/png;base64,{{ original_image }}" alt="Оригинал">
                </div>
                <div class="image-box">
                    <b>✨ После свёртки:</b><br>
                    <img src="data:image/png;base64,{{ result_image }}" alt="Результат">
                </div>
            </div>
            <div class="result">
                <b>Использованное ядро:</b>
                <pre>{{ kernel_used }}</pre>
            </div>
            {% endif %}
        </div>

        <div id="mode-multilayer" style="display:none">
            <h3>🏗️ Конструктор свёрточной сети</h3>
            <p>Добавляйте слои с разными ядрами. <b>ReLU</b> (обнуление отрицательных значений) применяется автоматически между слоями!</p>

            <form method="post" action="/multilayer" enctype="multipart/form-data">
                <b>Загрузите изображение:</b><br>
                <input type="file" name="image" accept="image/*" required><br><br>

                <div id="layers-container">
                    <div class="layer-card" data-layer="1" data-type="conv">
                        <h4>📌 Слой 1 (Свёртка)</h4>
                        <label>Название слоя:</label>
                        <input type="text" name="name_1" value="Границы" style="width: 200px;"><br><br>
                        <label>Ядро (матрица 3x3):</label><br>
                        <textarea name="kernel_1" rows="2" cols="40">[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]</textarea><br>
                        <button type="button" onclick="removeLayer(1)" style="background:#dc3545; margin-top:10px;">❌ Удалить слой</button>
                    </div>
                    <div class="layer-card" data-layer="2" data-type="conv">
                        <h4>📌 Слой 2 (Свёртка)</h4>
                        <label>Название слоя:</label>
                        <input type="text" name="name_2" value="Размытие" style="width: 200px;"><br><br>
                        <label>Ядро (матрица 3x3):</label><br>
                        <textarea name="kernel_2" rows="2" cols="40">[[1,1,1],[1,1,1],[1,1,1]]/9</textarea><br>
                        <button type="button" onclick="removeLayer(2)" style="background:#dc3545; margin-top:10px;">❌ Удалить слой</button>
                    </div>
                </div>

                <div style="margin: 10px 0;">
                    <button type="button" onclick="addLayer('conv')">➕ Добавить свёрточный слой</button>
                    <button type="button" onclick="addLayer('pool')">📐 Добавить MaxPooling слой</button>
                </div>
                <button type="button" onclick="loadArchitecture('edges')" style="background:#28a745;">🔲 Границы → Размытие → Резкость</button>
                <button type="button" onclick="loadArchitecture('blur')" style="background:#28a745;">📷 Только размытие (2 слоя)</button>
                <button type="button" onclick="loadArchitecture('sharpen')" style="background:#28a745;">🔪 Только резкость</button>
                <button type="button" onclick="loadArchitecture('vgg_block')" style="background:#17a2b8;">🏗️ VGG блок (Свёртка→Свёртка→Pooling)</button>

                <br><br>
                <button type="submit">🚀 Запустить нейросеть</button>
            </form>

            {% if multilayer_original %}
            <div class="flex-container" style="margin-top: 30px;">
                <div class="image-box">
                    <b>📸 Оригинал:</b><br>
                    <img src="data:image/png;base64,{{ multilayer_original }}" alt="Оригинал">
                </div>
                {% for res in multilayer_results %}
                <div class="image-box">
                    <b>{{ res.name }}<br>(после слоя {{ res.layer }})</b><br>
                    <img src="data:image/png;base64,{{ res.image_b64 }}" alt="После слоя {{ res.layer }}">
                </div>
                {% endfor %}
            </div>
            {% endif %}

            {% if multilayer_error %}
            <div class="result error">
                <b>Ошибка:</b> {{ multilayer_error }}
            </div>
            {% endif %}
        </div>

        {% if error %}
        <div class="result error">
            <b>Ошибка:</b> {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>
'''


def convolution_matrix(matrix, kernel, stride=1, padding=0):
    rows, cols = len(matrix), len(matrix[0])
    krows, kcols = len(kernel), len(kernel[0])

    if padding > 0:
        new_rows = rows + 2 * padding
        new_cols = cols + 2 * padding
        padded = [[0] * new_cols for _ in range(new_rows)]
        for i in range(rows):
            for j in range(cols):
                padded[i + padding][j + padding] = matrix[i][j]
        matrix = padded
        rows, cols = new_rows, new_cols

    res_rows = (rows - krows) // stride + 1
    res_cols = (cols - kcols) // stride + 1

    if res_rows <= 0 or res_cols <= 0:
        raise ValueError("Ядро слишком большое")

    res = [[0] * res_cols for _ in range(res_rows)]

    for i in range(0, res_rows * stride, stride):
        for j in range(0, res_cols * stride, stride):
            total = 0
            for ki in range(krows):
                for kj in range(kcols):
                    total += matrix[i + ki][j + kj] * kernel[ki][kj]
            res[i // stride][j // stride] = total

    return res


def parse_kernel(kernel_str):
    kernel_str = kernel_str.strip()
    if '/' in kernel_str:
        parts = kernel_str.split('/')
        kernel = json.loads(parts[0])
        divisor = float(parts[1])
    else:
        kernel = json.loads(kernel_str)
        divisor = 1

    for i in range(len(kernel)):
        for j in range(len(kernel[0])):
            kernel[i][j] = kernel[i][j] / divisor

    return kernel


def apply_relu(matrix):
    return [[max(0, val) for val in row] for row in matrix]


def normalize_matrix_for_display(matrix):
    min_val = min(min(row) for row in matrix)
    max_val = max(max(row) for row in matrix)

    if max_val - min_val > 0:
        normalized = [[int((val - min_val) * 255 / (max_val - min_val)) for val in row] for row in matrix]
    else:
        normalized = [[128 for val in row] for row in matrix]

    return normalized


def max_pooling(matrix, pool_size=2, stride=2):
    rows, cols = len(matrix), len(matrix[0])
    res_rows = (rows - pool_size) // stride + 1
    res_cols = (cols - pool_size) // stride + 1

    if res_rows <= 0 or res_cols <= 0:
        raise ValueError("Pooling размер слишком большой для данной матрицы")

    result = [[0] * res_cols for _ in range(res_rows)]

    for i in range(0, res_rows * stride, stride):
        for j in range(0, res_cols * stride, stride):
            max_val = matrix[i][j]
            for pi in range(pool_size):
                for pj in range(pool_size):
                    if matrix[i + pi][j + pj] > max_val:
                        max_val = matrix[i + pi][j + pj]
            result[i // stride][j // stride] = max_val

    return result


@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE,
                                  matrix_a="[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]",
                                  matrix_b="[[1,0,0],[0,1,0],[0,0,1]]",
                                  padding=0, stride=1)


@app.route('/matrix', methods=['POST'])
def matrix_mode():
    try:
        matrix_a_str = request.form.get('matrix_a')
        matrix_b_str = request.form.get('matrix_b')
        padding = int(request.form.get('padding', 0))
        stride = int(request.form.get('stride', 1))

        a = json.loads(matrix_a_str)
        b = json.loads(matrix_b_str)
        res = convolution_matrix(a, b, stride, padding)

        return render_template_string(HTML_TEMPLATE,
                                      matrix_a=matrix_a_str,
                                      matrix_b=matrix_b_str,
                                      padding=padding,
                                      stride=stride,
                                      matrix_a_display=a,
                                      matrix_b_display=b,
                                      result_matrix=res,
                                      a_rows=len(a), a_cols=len(a[0]),
                                      b_rows=len(b), b_cols=len(b[0]),
                                      res_rows=len(res), res_cols=len(res[0]))
    except Exception as e:
        return render_template_string(HTML_TEMPLATE,
                                      error=str(e),
                                      matrix_a=request.form.get('matrix_a', ''),
                                      matrix_b=request.form.get('matrix_b', ''),
                                      padding=0, stride=1)


@app.route('/image', methods=['POST'])
def image_mode():
    try:
        file = request.files['image']
        img = Image.open(file.stream).convert('L')

        kernel_str = request.form.get('custom_kernel', '[[0,0,0],[0,1,0],[0,0,0]]')
        kernel = parse_kernel(kernel_str)

        img_array = [[float(img.getpixel((x, y))) for x in range(img.width)] for y in range(img.height)]

        result_array = convolution_matrix(img_array, kernel)

        min_val = min(min(row) for row in result_array)
        max_val = max(max(row) for row in result_array)

        if max_val - min_val > 0:
            result_array = [[int((val - min_val) * 255 / (max_val - min_val)) for val in row] for row in result_array]
        else:
            result_array = [[128 for val in row] for row in result_array]

        result_img = Image.new('L', (len(result_array[0]), len(result_array)))
        for y in range(len(result_array)):
            for x in range(len(result_array[0])):
                result_img.putpixel((x, y), int(result_array[y][x]))

        original_buf = io.BytesIO()
        img.save(original_buf, format='PNG')
        original_b64 = base64.b64encode(original_buf.getvalue()).decode()

        result_buf = io.BytesIO()
        result_img.save(result_buf, format='PNG')
        result_b64 = base64.b64encode(result_buf.getvalue()).decode()

        return render_template_string(HTML_TEMPLATE,
                                      original_image=original_b64,
                                      result_image=result_b64,
                                      kernel_used=json.dumps(kernel, indent=2))
    except Exception as e:
        return render_template_string(HTML_TEMPLATE,
                                      error=f"Ошибка обработки: {str(e)}")


@app.route('/multilayer', methods=['POST'])
def multilayer_mode():
    try:
        file = request.files['image']
        img = Image.open(file.stream).convert('L')
        img_array = [[float(img.getpixel((x, y))) for x in range(img.width)] for y in range(img.height)]

        layers = []
        layer_keys = {}
        for key in request.form:
            if key.startswith('name_'):
                layer_num = key.split('_')[1]
                layer_keys[layer_num] = request.form[key]

        for layer_num in sorted(layer_keys.keys(), key=int):
            layer_name = layer_keys[layer_num]

            if request.form.get(f'is_pool_{layer_num}'):
                layers.append({
                    'type': 'pool',
                    'name': layer_name
                })
            else:
                kernel_str = request.form.get(f'kernel_{layer_num}', '')
                if kernel_str and kernel_str.strip():
                    kernel = parse_kernel(kernel_str)
                    layers.append({
                        'type': 'conv',
                        'name': layer_name,
                        'kernel': kernel
                    })

        if not layers:
            layers = [
                {'type': 'conv', 'name': 'Границы', 'kernel': parse_kernel('[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]')},
                {'type': 'conv', 'name': 'Размытие', 'kernel': parse_kernel('[[1,1,1],[1,1,1],[1,1,1]]/9')},
                {'type': 'conv', 'name': 'Резкость', 'kernel': parse_kernel('[[0,-1,0],[-1,5,-1],[0,-1,0]]')}
            ]

        results = []
        current = img_array

        for i, layer in enumerate(layers):
            if layer['type'] == 'conv':

                conv_result = convolution_matrix(current, layer['kernel'])
                current = apply_relu(conv_result)
                layer_display_name = f"{layer['name']} (Свёртка+ReLU)"
            else:

                current = max_pooling(current)
                layer_display_name = f"{layer['name']} (MaxPooling)"

            normalized = normalize_matrix_for_display(current)

            res_img = Image.new('L', (len(normalized[0]), len(normalized)))
            for y in range(len(normalized)):
                for x in range(len(normalized[0])):
                    res_img.putpixel((x, y), int(normalized[y][x]))

            buf = io.BytesIO()
            res_img.save(buf, format='PNG')
            results.append({
                'name': layer_display_name,
                'layer': i + 1,
                'image_b64': base64.b64encode(buf.getvalue()).decode()
            })

        original_buf = io.BytesIO()
        img.save(original_buf, format='PNG')
        original_b64 = base64.b64encode(original_buf.getvalue()).decode()

        return render_template_string(HTML_TEMPLATE,
                                      multilayer_original=original_b64,
                                      multilayer_results=results)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE,
                                      multilayer_error=f"Ошибка: {str(e)}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
