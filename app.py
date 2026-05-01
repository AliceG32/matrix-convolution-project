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
    </style>
    <script>
        function switchMode(mode) {
            document.getElementById('mode-matrix').style.display = mode === 'matrix' ? 'block' : 'none';
            document.getElementById('mode-image').style.display = mode === 'image' ? 'block' : 'none';
            document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelector(`.mode-btn[onclick="switchMode('${mode}')"]`).classList.add('active');
        }

        function setKernel(kernel) {
            document.getElementById('custom_kernel').value = kernel;
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)